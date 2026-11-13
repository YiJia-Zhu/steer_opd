#!/usr/bin/env python3
"""Plot saved probe/final overlaps and actual masks; paired comparison is default."""

import argparse
import csv
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator
import torch

from steer_opd.resumable_core import ROOT, atomic_json, read_json
from steer_opd.rollout_topk import load_rollout_topk


def overlap(student_ids, teacher_ids):
    return (student_ids[:, :, None] == teacher_ids[:, None, :]).any(-1).sum(-1).double() / student_ids.shape[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--comparison", action="store_true",
                        help="Overlay the saved final-policy overlap and add a per-token difference panel")
    parser.add_argument("--probe-only", action="store_false", dest="comparison",
                        help="Explicitly request the historical single-curve view")
    parser.set_defaults(comparison=True)
    args = parser.parse_args()
    rollouts = (ROOT / args.rollouts).resolve()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to(ROOT.resolve()):
        raise ValueError("Plot artifacts must remain within this project")
    job_path = rollouts / "job.json"
    job = read_json(job_path)["job"]
    if not job.get("method"):
        raise ValueError("This rollout has no paired method/probe records; do not invent a no-steering curve")
    if not 1 <= args.count <= len(job["requests"]):
        raise ValueError("Invalid number of samples")
    output.mkdir(parents=True, exist_ok=True)
    threshold = job["method"]["overlap_threshold"] if job["method"]["gate"] == "overlap" else None
    metadata = {"source_job": str(job_path.relative_to(ROOT)),
        "source_job_sha256": hashlib.sha256(job_path.read_bytes()).hexdigest(),
        "selection": "first requests in original generation manifest order; no filtering",
        "method": job["method"], "overlap_threshold": threshold,
        "x_axis": "0-based response token index, including any terminal token",
        "y_axis": "current-step no-intervention probe Overlap@16 on actual controlled history",
        "scatter": "actual saved steering_mask, checked against threshold decisions",
        "smoothing": False, "new_model_inference": False,
        "comparison": args.comparison,
        "comparison_scope": "same actual controlled history; current-step probe vs actual final policy",
        "independent_whole_trajectory_no_steering_control": False,
        "samples": []}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "svg.fonttype": "none", "pdf.fonttype": 42, "path.simplify": False})
    pdf_name = f"first{args.count}_token_overlap.pdf"
    method = job["method"]
    method_label = (f"{method['direction'] if job.get('vector') else 'No SV'} / {method['gate']} / layers "
                    + "+".join(map(str, method["layers"])) + f" / rho {method['rho']:g}")
    with PdfPages(output / pdf_name) as pdf:
        for ordinal, request in enumerate(job["requests"][:args.count], 1):
            key = request["key"]
            row_path = rollouts / f"{key}.json"
            row = read_json(row_path)
            scores = load_rollout_topk(rollouts, row, job["save_rollout_top_k"], job["support_size"])
            if "probe_topk_ids" not in scores or "teacher_topk_ids" not in scores:
                raise ValueError(f"{key}: missing saved paired probe/teacher Top16; cannot plot a counterfactual")
            probe, final, teacher = (scores[k].long() for k in ("probe_topk_ids", "ids", "teacher_topk_ids"))
            n = len(row["token_ids"])
            if probe.shape != final.shape or final.shape != teacher.shape or len(probe) != n:
                raise ValueError(f"Token record shape mismatch: {key}")
            before, after = overlap(probe, teacher), overlap(final, teacher)
            actual = torch.tensor(row["steering_mask"], dtype=torch.bool)
            requested = before < threshold if threshold is not None else actual.clone()
            if threshold is not None:
                zero = torch.tensor([metric.get("zero_vector_no_intervention", False)
                                     for metric in row.get("gate_metrics", [{}]*n)], dtype=torch.bool)
                if (actual & ~requested).any() or (job.get("vector") and (requested & ~actual & ~zero).any()):
                    raise ValueError(f"Actual mask differs from overlap threshold: {key}")
            if not torch.equal(before[~actual], after[~actual]):
                raise ValueError(f"Probe differs from final policy without intervention: {key}")
            stem = f"sample_{ordinal:02d}_{key}"
            csv_path = output / f"{stem}.csv"
            with csv_path.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["token_idx", "token_id", "probe_overlap16", "final_overlap16", "delta_overlap_pp",
                                 "requested_gate", "actual_steer", "overlap_threshold"])
                for i in range(n):
                    writer.writerow([i, row["token_ids"][i], float(before[i]), float(after[i]), float(100*(after[i]-before[i])),
                                     int(requested[i]), int(actual[i]), threshold])
            x = torch.arange(n).numpy()
            y, mask = before.numpy(), actual.numpy()
            delta = after-before
            if args.comparison:
                fig, (ax, delta_ax) = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                    gridspec_kw={"height_ratios": [4, 1.25], "hspace": .12})
                fig.subplots_adjust(left=.075, right=.985, bottom=.14, top=.77)
            else:
                fig, ax = plt.subplots(figsize=(14, 4.8))
                fig.subplots_adjust(left=.075, right=.985, bottom=.17, top=.79)
            fig.suptitle(f"Sample {ordinal} | {key} | {row['id']} | seed {row['seed']}",
                         x=.075, y=.97, ha="left", fontsize=15, fontweight="bold")
            fig.text(.075, .925 if args.comparison else .885, f"{method_label}   |   "
                     f"{n:,} tokens, {int(actual.sum()):,} steered ({100*float(actual.double().mean()):.2f}%)",
                     fontsize=10, color="#475569")
            if args.comparison:
                fig.text(.075, .885, f"Mean probe: {100*float(before.mean()):.6f}%   |   "
                         f"Mean actual: {100*float(after.mean()):.6f}%   |   "
                         f"Mean delta: {100*float(delta.mean()):+.6f} pp   |   "
                         f"Changed positions: {int((delta!=0).sum())}", fontsize=10, color="#334155")
            ax.plot(x, y, color="#2563A6", lw=.9 if n < 1000 else .65,
                    alpha=.85, label="No injection at current step (probe)" if args.comparison else "Pre-steering probe overlap", zorder=2)
            if args.comparison:
                ax.plot(x, after.numpy(), color="#E99B28", lw=1.05 if n<1000 else .75,
                        ls=(0, (5, 3)), alpha=.9, label="Actual policy after gate", zorder=3)
            ax.scatter(x[mask], y[mask], s=35 if n < 100 else 20, color="#E34B36",
                       edgecolors="white", linewidths=.35, label="Actual steer positions", zorder=4)
            if threshold is not None:
                ax.axhline(threshold, color="#6B7280", ls=(0, (5, 4)), lw=1.1,
                           label=f"Gate threshold (~{threshold:.3g})", zorder=1)
            ax.set(ylabel="Overlap@16", ylim=(-.035, 1.035))
            if not args.comparison:
                ax.set_xlabel("Response token idx (0-based)")
            ax.set_xlim(-.5, n-.5)
            ax.set_yticks([0, .25, .375, .5, .75, 1.0])
            ax.xaxis.set_major_locator(MaxNLocator(nbins=12, integer=True))
            if n <= 20:
                ax.set_xticks(x)
            ax.grid(axis="y", color="#E2E8F0", lw=.7)
            ax.legend(loc="lower left", bbox_to_anchor=(-.008, 1.005), ncol=2 if args.comparison else 3,
                      frameon=False, fontsize=10, handlelength=2.5, columnspacing=2)
            if args.comparison:
                difference = delta.numpy()*100
                nonzero = difference!=0
                delta_ax.axhline(0, color="#94A3B8", lw=.8)
                delta_ax.plot(x, difference, color="#7854A6", lw=.8, zorder=2)
                delta_ax.scatter(x[nonzero], difference[nonzero], color="#7854A6", s=17, zorder=3)
                limit=max(6.25, float(abs(difference).max()))
                delta_ax.set_ylim(-1.2*limit, 1.2*limit)
                delta_ax.set_yticks([-limit, 0, limit])
                delta_ax.set(ylabel="Delta (pp)", xlabel="Response token idx (0-based)")
                delta_ax.grid(axis="y", color="#E2E8F0", lw=.7)
                if not nonzero.any():
                    delta_ax.text(.5, .8, "All token overlaps are unchanged", transform=delta_ax.transAxes,
                                  ha="center", fontsize=10, color="#64748B")
            fig.text(.075, .035, "Saved trajectory; raw per-token values. Prior controlled history is held fixed.",
                     fontsize=9, color="#64748B")
            fig.savefig(output / f"{stem}.png", dpi=200, facecolor="white")
            fig.savefig(output / f"{stem}.svg", facecolor="white")
            pdf.savefig(fig, facecolor="white")
            plt.close(fig)
            sample = {"ordinal": ordinal, "key": key, "question_id": row["id"], "seed": row["seed"],
                "tokens": n, "steered_tokens": int(actual.sum()), "source_row": str(row_path.relative_to(ROOT)),
                "source_row_sha256": hashlib.sha256(row_path.read_bytes()).hexdigest(),
                "source_topk_sha256": row["rollout_topk_hash"],
                "png": f"{stem}.png", "svg": f"{stem}.svg", "csv": csv_path.name,
                "probe_mean_percent": 100*float(before.mean()),
                "final_mean_percent": 100*float(after.mean()),
                "delta_mean_pp": 100*float(delta.mean()),
                "changed_positions": int((delta!=0).sum()),
                "improved_positions": int((delta>0).sum()),
                "decreased_positions": int((delta<0).sum()),
                "mask_threshold_mismatches": 0}
            metadata["samples"].append(sample)
            print(sample, flush=True)
    atomic_json(output / "manifest.json", metadata)
    links = "\n".join(f"- {s['key']}（{s['tokens']} token，{s['steered_tokens']}个注入点）："
        f"[PNG]({s['png']}) / [SVG]({s['svg']}) / [逐token数据]({s['csv']})" for s in metadata["samples"])
    sample_description = "；".join(f"{s['key']}（题目{s['question_id']}，seed{s['seed']}）" for s in metadata["samples"])
    comparison_note = ("新增橙色虚线为保存的正式策略overlap，蓝线为当前步不注入的probe；下方紫色差值为（正式−probe），单位百分点。\n"
        "两条线保留相同的实际受控历史，只切换当前步注入。前面的历史仍可能受SV影响，不能把probe称为整段无steer独立生成的对照。\n") if args.comparison else ""
    stats = "\n".join(f"| {s['key']} | {s['probe_mean_percent']:.6f}% | {s['final_mean_percent']:.6f}% | {s['delta_mean_pp']:+.6f} | {s['changed_positions']} |" for s in metadata["samples"])
    threshold_note = (f"保存的阈值为{threshold!r}，图中四舍五入显示{threshold:.3g}；阈值判断与实际mask逐位置核验（零向量/无SV明确排除实际注入）。\n"
                      if threshold is not None else "本方法不使用overlap阈值判断，图中不绘制overlap门槛线，散点来自实际保存的mask。\n")
    (output / "README.md").write_text(f"# 前{args.count}条回答的逐token overlap与门控位置\n\n"
        + f"按原生成清单顺序选择：{sample_description}。\n"
        + comparison_note
        + "\n| 样本 | 当前步无注入probe均值 | 正式策略均值 | 增量（百分点） | overlap变化位置数 |\n|---|---:|---:|---:|---:|\n"
        + stats + "\n\n"
        +
        "蓝色折线是保留实际受控历史、当前步尚未注入的probe Overlap@16；红色散点标记原始日志中的实际注入位置。\n"
        "横轴从首个回答token的0开始，包含最后的终止token或长度上限位置；没有平滑、抽样或裁掉长回答。\n"
        + threshold_note + "\n"
        + links + f"\n\n[{args.count}页PDF]({pdf_name}) / [来源与检查记录](manifest.json)\n\n"
        "CSV同时保存probe和正式分布的overlap及实际mask，图中折线采用门控所依据的probe值。\n"
        "这些图展示当前未更新学生在同一实际受控历史上的门控记录，不表示训练后关闭steering的收益。未调用GPU或重新生成回答。\n")


if __name__ == "__main__":
    main()
