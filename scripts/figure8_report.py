#!/usr/bin/env python3
"""CPU-only original-grader evaluation, durable records, and experiment plots."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from pathlib import Path
import signal
import time

from steer_opd.resumable_core import ROOT, atomic_json, read_json


def plot_curves(run, rows, groups):
    """Optional presentation: failures must not invalidate saved measurements."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    try:
        for group in groups:
            selected = [r for r in rows if r["group"] == group]
            for ax, key, label in zip(axes, ["accuracy", "overlap16"], ["AIME2025 avg@16", "No-hook Overlap@16"]):
                valid = [r for r in selected if r[key] is not None]
                if valid:
                    ax.plot([r["step"] for r in valid], [100 * r[key] for r in valid], marker="o", label=group)
                ax.set_xlabel("Optimizer step")
                ax.set_ylabel(label + " (%)")
                ax.grid(alpha=0.25)
                if ax.lines:
                    ax.legend()
        figure.tight_layout()
        figure.savefig(run / "curves.png", dpi=180)
    finally:
        plt.close(figure)


def grade_one(item):
    key, row, output = item
    from steer_opd.math_data import grade_rethinking_math

    def timeout(signum, frame):
        raise TimeoutError("Answer grader exceeded 20 seconds")

    signal.signal(signal.SIGALRM, timeout)
    try:
        signal.alarm(20)
        result = grade_rethinking_math(row["answer"], row["text"])
    except Exception as exc:
        result = {"correct": False, "grading_status": "exception", "error": repr(exc)}
    finally:
        signal.alarm(0)
    record = {"key": key, "id": row["id"], "answer": row["answer"], **result,
              "tokens": len(row["token_ids"]), "finish_reason": row["finish_reason"]}
    atomic_json(Path(output) / f"{key}.json", record)
    return record


def grade(path):
    job = read_json(path / "answers/job.json")["job"]
    output = path / "grades"
    pending = [(r["key"], read_json(path / "answers" / f"{r['key']}.json"), str(output))
               for r in job["requests"] if not (output / f"{r['key']}.json").exists()]
    if pending:
        with ProcessPoolExecutor(max_workers=8) as pool:
            list(pool.map(grade_one, pending))
    records = [read_json(output / f"{r['key']}.json") for r in job["requests"]]
    per_question = {}
    for row in records:
        per_question.setdefault(row["id"], []).append(bool(row["correct"]))
    sample_counts = {len(values) for values in per_question.values()}
    metric = f"avg@{next(iter(sample_counts))}" if len(sample_counts) == 1 else "mean_accuracy_variable_samples"
    result = {"accuracy": sum(r["correct"] for r in records) / len(records),
              "samples": len(records), "questions": len(per_question), "metric": metric,
              "per_question": {k: sum(v) / len(v) for k, v in per_question.items()},
              "mean_tokens": sum(r["tokens"] for r in records) / len(records),
              "truncated_fraction": sum(r["finish_reason"] == "length" for r in records) / len(records),
              "grading_exceptions": sum(r["grading_status"] == "exception" for r in records),
              "student_steering": False, "grader": "original_rethinking_grade_answer_verl"}
    if result["grading_exceptions"]:
        atomic_json(path / "grading_errors.json", result)
        raise RuntimeError("Grading exceptions require review; not silently counted as valid accuracy")
    atomic_json(path / "accuracy.json", result)


def report(cfg):
    run = ROOT / cfg["run_dir"]
    rows, progress = [], {}
    for group in cfg["groups"]:
        metrics = sorted((run / group / "steps").glob("*/train_metrics.json"))
        latest_path = run / group / "latest.json"
        latest = read_json(latest_path)["step"] if latest_path.exists() else 0
        progress[group] = max([latest, *[read_json(p)["step"] for p in metrics]])
        for step in sorted(set(cfg["eval_steps"] + cfg.get("dev_eval_steps", []) + [0])):
            path = run / "evaluation" / ("base" if step == 0 else f"{group}_{step:04d}")
            if not (path / "accuracy.json").exists() and not (path / "overlap.json").exists():
                continue
            accuracy = read_json(path / "accuracy.json") if (path / "accuracy.json").exists() else {}
            overlap = read_json(path / "overlap.json") if (path / "overlap.json").exists() else {}
            rows.append({"group": group, "step": step, "accuracy": accuracy.get("accuracy"),
                         "overlap16": overlap.get("overlap16"), "mean_tokens": accuracy.get("mean_tokens"),
                         "truncated_fraction": accuracy.get("truncated_fraction")})
    result = {"completed_steps": progress, "planned_steps": cfg["steps"], "evaluation": rows}
    atomic_json(run / "results.json", result)
    if rows:
        with (run / "curves.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = ["# 首轮OPD实验进度", "", f"已完成更新：{progress}；目标每组{cfg['steps']}步。", "",
             "主评测关闭steering；论文SFT数字仅作近似参照。", "",
             "| 组 | step | AIME avg@16 | 无hook Overlap@16 |", "|---|---:|---:|---:|"]
    for row in rows:
        overlap = "待评" if row["overlap16"] is None else f"{100 * row['overlap16']:.4f}%"
        accuracy = "待评" if row["accuracy"] is None else f"{100 * row['accuracy']:.2f}%"
        lines.append(f"| {row['group']} | {row['step']} | {accuracy} | {overlap} |")
    lines += ["", "恢复：在项目根source scripts/runtime_env.sh后，运行", "",
              "```bash", '"$STEER_OPD_PYTHON" scripts/run_figure8_campaign.py --config configs/figure8_first_run.json',
              "```", "", "每次自动重新检查空闲GPU，优先单卡；不会终止其它进程。"]
    (run / "README.md").write_text("\n".join(lines) + "\n")
    if rows:
        try:
            plot_curves(run, rows, cfg["groups"])
        except Exception as exc:
            atomic_json(run / "plot_status.json", {"status": "deferred", "error": repr(exc),
                        "measurements_saved": True, "updated_at": time.time()})
            print(f"Plot deferred; measurements and progress saved: {exc}", flush=True)
        else:
            atomic_json(run / "plot_status.json", {"status": "complete", "updated_at": time.time()})
    print(result["completed_steps"], flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/figure8_first_run.json")
    parser.add_argument("--grade", type=Path)
    parser.add_argument("--grade-only", action="store_true")
    args = parser.parse_args()
    if args.grade:
        grade(args.grade)
        if args.grade_only:
            return
    report(read_json(args.config))


if __name__ == "__main__":
    main()
