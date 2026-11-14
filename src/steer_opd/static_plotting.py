"""CPU plots: whole-source no-hook baseline versus static intervention."""

import csv
import hashlib

import torch

from steer_opd.resumable_core import atomic_json


def plot_comparisons(rows, baseline, candidate, output, *, first=3, requested_keys=None, metric="overlap"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"baseline": "whole_source_no_hook_fixed_prefix", "candidate": "fixed_mask_same_source_prefix",
                "current_step_probe": False, "selection": "first rows in source manifest order", "metric": metric, "rows": []}
    by_key = {r.get("key", r["id"]): r for r in rows}
    requested_keys = requested_keys if requested_keys is not None else [r.get("key", r["id"]) for r in rows[:first]]
    manifest["requested_keys"] = list(requested_keys)
    manifest["unavailable_keys"] = [k for k in requested_keys if k not in by_key]
    with PdfPages(output / "first3.pdf") as pdf:
        for row in [by_key[k] for k in requested_keys if k in by_key]:
            key = row.get("key", row["id"])
            a, b = baseline[key], candidate[key]
            before, after = a[metric].double().cpu().numpy(), b[metric].double().cpu().numpy()
            mask = b["steering_mask"].bool().cpu().numpy()
            valid = b.get("alignment_mask", torch.ones(len(before), dtype=torch.bool)).cpu().numpy()
            requested = b["requested_mask"].bool().cpu().numpy()
            x = list(range(len(before)))
            with (output / f"{key}.csv").open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["prediction_idx", f"baseline_{metric}", f"candidate_{metric}", "delta_pp", "requested", "actual", "metric_valid"])
                writer.writerows((i, before[i], after[i], 100*(after[i]-before[i]), int(requested[i]), int(mask[i]), int(valid[i])) for i in x)
            fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
            axes[0].plot(x, before, color="#2563eb", lw=.9, label="Whole-source no SV")
            axes[0].plot(x, after, color="#ea580c", lw=.9, label="Fixed-mask intervention")
            axes[0].scatter([i for i in x if mask[i] and valid[i]], after[mask & valid], color="#dc2626", s=10, label="Actual steer")
            if (mask & ~valid).any():
                axes[0].scatter([i for i in x if mask[i] and not valid[i]], [.02]*int((mask & ~valid).sum()),
                    transform=axes[0].get_xaxis_transform(), marker="|", color="#dc2626", label="Steer, metric unavailable")
            axes[0].set(ylabel="P_overlap" if metric == "P_overlap" else "Overlap@16", ylim=(-.02, 1.02), title=f"{key}: same source prefixes, zero updates")
            axes[0].legend(loc="best")
            axes[1].plot(x, 100*(after-before), color="#7c3aed", lw=.8)
            axes[1].axhline(0, color="gray", lw=.5)
            axes[1].set(xlabel="Response prediction index", ylabel="Delta (pp)")
            fig.tight_layout()
            for suffix in ["png", "svg"]:
                fig.savefig(output / f"{key}.{suffix}", dpi=160)
            pdf.savefig(fig)
            plt.close(fig)
            manifest["rows"].append({"key": key, "mask_id": b["mask_id"], "binding": b["binding"],
                                     "positions": len(x), "steered": int(mask.sum()), "metric_valid_positions": int(valid.sum()),
                                     "metric": b.get("metric", "native_overlap_at_16")})
    manifest["files"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.suffix in (".png", ".svg", ".csv", ".pdf")}
    atomic_json(output / "manifest.json", manifest)


def plot_dual_comparisons(rows, baseline, candidate, output, *, requested_keys=None):
    for metric in ("overlap", "P_overlap"):
        plot_comparisons(rows, baseline, candidate, output / metric,
                         requested_keys=requested_keys, metric=metric)
