"""
Plots accuracy vs an independent variable -- n_circles (the FINST/MOT
capacity-limit signal this whole project is testing for) by default, but
any swept column in results.csv (fps, speed_px_s) via --x-axis -- from a
run_comprehension_batch.py results.csv, one color-coded series per model,
aggregated (mean +/- SEM) across seeds.

Works for any sweep shape: a single-value pilot (one seed range at one x
value, to get a stable accuracy estimate) renders as one point per model
with error bars from seed-to-seed variance; a multi-value sweep renders the
full curve.

Reads from and writes back to results/<batch_id>/ (not data/<batch_id>/) --
run_comprehension_batch.py puts results.csv there specifically because
results/ is git-tracked and data/ is gitignored, so the plot lands alongside
it for the same reason.

Usage:
    uv run python scripts/plot_comprehension_results.py results/<batch_id>
    uv run python scripts/plot_comprehension_results.py results/<batch_id> --x-axis fps
    uv run python scripts/plot_comprehension_results.py results/<batch_id> --x-axis speed_px_s
"""

import argparse
import csv
import os
import statistics
from collections import defaultdict

import matplotlib.pyplot as plt

# Categorical palette slots 1-3 (fixed order; validated for adjacent CVD
# separation and normal-vision floor at data-viz skill's references/palette.md).
MODEL_COLORS = ["#2a78d6", "#008300", "#e87ba4"]  # blue, green, magenta
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"


def load_rows(results_csv: str) -> list[dict]:
    with open(results_csv) as f:
        return list(csv.DictReader(f))


def aggregate(rows: list[dict], x_axis: str) -> dict:
    """Groups trials by (model, x_axis value) and reduces each group to
    mean/SEM for exact-match rate and mean recall. Trials with a recorded
    `error` (failed API calls) are excluded rather than counted as wrong."""
    groups = defaultdict(list)
    for row in rows:
        if row["error"]:
            continue
        key = (row["model"], float(row[x_axis]))
        groups[key].append({
            "exact_match": row["exact_match"] == "True",
            "recall": float(row["recall"]),
        })

    summary = {}
    for key, trials in groups.items():
        n = len(trials)
        exact = [t["exact_match"] for t in trials]
        recall = [t["recall"] for t in trials]
        summary[key] = {
            "n": n,
            "exact_match_rate": sum(exact) / n,
            "exact_match_sem": (statistics.pstdev(exact) / (n ** 0.5)) if n > 1 else 0.0,
            "recall_mean": sum(recall) / n,
            "recall_sem": (statistics.pstdev(recall) / (n ** 0.5)) if n > 1 else 0.0,
        }
    return summary


def plot(summary: dict, out_path: str, x_axis: str):
    models = sorted({model for model, _ in summary})
    all_x = sorted({x for _, x in summary})
    fig, (ax_exact, ax_recall) = plt.subplots(1, 2, figsize=(11, 4.5), facecolor="#fcfcfb")

    for ax in (ax_exact, ax_recall):
        ax.set_facecolor("#fcfcfb")
        ax.grid(True, color=GRIDLINE, linewidth=1, zorder=0)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(BASELINE)
        ax.tick_params(colors=INK_MUTED)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks(all_x)
        pad = (max(all_x) - min(all_x)) * 0.08 if len(all_x) > 1 else 1.0
        ax.set_xlim(min(all_x) - pad, max(all_x) + pad)

    for i, model in enumerate(models):
        color = MODEL_COLORS[i % len(MODEL_COLORS)]
        points = sorted((n, s) for (m, n), s in summary.items() if m == model)
        xs = [n for n, _ in points]
        n_trials = [s["n"] for _, s in points]
        label = f"{model} (n={min(n_trials)})" if len(set(n_trials)) == 1 else model

        ax_exact.errorbar(
            xs, [s["exact_match_rate"] for _, s in points],
            yerr=[s["exact_match_sem"] for _, s in points],
            marker="o", markersize=8, linewidth=2, capsize=3,
            color=color, label=label, zorder=3,
        )
        ax_recall.errorbar(
            xs, [s["recall_mean"] for _, s in points],
            yerr=[s["recall_sem"] for _, s in points],
            marker="o", markersize=8, linewidth=2, capsize=3,
            color=color, label=label, zorder=3,
        )

    for ax, title, ylabel in (
        (ax_exact, "Exact identity match", "Fraction of trials correct"),
        (ax_recall, "Recall (partial credit)", "Mean recall"),
    ):
        ax.set_xlabel(x_axis, color=INK_SECONDARY)
        ax.set_ylabel(ylabel, color=INK_SECONDARY)
        ax.set_title(title, color=INK_PRIMARY, fontsize=11)

    handles, labels = ax_exact.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02),
        ncol=min(len(models), 3), frameon=False, labelcolor=INK_PRIMARY, fontsize=9,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "batch_dir",
        help="Path to a results/<batch_id> directory produced by run_comprehension_batch.py",
    )
    parser.add_argument(
        "--x-axis", type=str, default="n_circles",
        help="results.csv column to use as the x-axis (n_circles, fps, or speed_px_s)",
    )
    args = parser.parse_args()

    results_csv = os.path.join(args.batch_dir, "results.csv")
    rows = load_rows(results_csv)
    summary = aggregate(rows, args.x_axis)

    out_name = "accuracy_plot.png" if args.x_axis == "n_circles" else f"accuracy_plot_{args.x_axis}.png"
    out_path = os.path.join(args.batch_dir, out_name)
    plot(summary, out_path, args.x_axis)


if __name__ == "__main__":
    main()
