"""
Summary figure for claude/2026_09/2026_09_09/TODO.md: locally-hosted
google/gemma-4-31b-it on debug_circular (rotation_deg=40) across a swept
num_frames budget (results/debug_circular/gemma_frame_sweep/results.csv,
produced by aggregate_gemma_frame_sweep.py). Line plot: num_frames on the
x-axis, accuracy (%) on the y-axis, one line (no native/stretched split --
this sweep has only one variant), with SEM error bars, mirroring
plot_angle_sweep.py's styling.

Marks num_frames=32 with a vertical reference line, since that's the fixed
frame budget OpenRouter's hosted endpoint was measured to silently truncate
every video to (see reference_gemma_fixed_frame_budget memory) -- the plot
directly visualizes whether accuracy at/above that point differs from below
it, and whether feeding more than 32 real frames (now possible locally, where
OpenRouter never allowed it) helps at all.

Usage:
    uv run python scripts/debug_circular/plot_gemma_frame_sweep.py
"""

import argparse
import csv
import json
import math
from collections import defaultdict

import matplotlib.pyplot as plt

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
LINE_COLOR = "#2a78d6"  # validated categorical slot 1
CAP_LINE_COLOR = "#eb6834"  # validated categorical slot 2


def _parse_bool(s: str) -> bool | None:
    if s == "True":
        return True
    if s == "False":
        return False
    return None


def accuracy_by_num_frames(results_csv: str) -> dict[int, tuple[float, float, int]]:
    """Returns {num_frames: (accuracy_pct, sem_pct, n)}."""
    totals = defaultdict(int)
    correct = defaultdict(int)
    with open(results_csv) as f:
        for row in csv.DictReader(f):
            key = int(row["num_frames"])
            totals[key] += 1
            if _parse_bool(row["predicted"]) == _parse_bool(row["probe_is_target"]):
                correct[key] += 1

    stats = {}
    for key, n in totals.items():
        p = correct[key] / n
        sem = math.sqrt(p * (1 - p) / n) * 100
        stats[key] = (p * 100, sem, n)
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", type=str, default="gemma_frame_sweep")
    parser.add_argument("--openrouter-cap", type=int, default=32)
    args = parser.parse_args()

    results_csv = f"results/debug_circular/{args.run_name}/results.csv"
    out_path = f"results/debug_circular/{args.run_name}/accuracy_by_n_frames.png"

    stats = accuracy_by_num_frames(results_csv)
    num_frames_values = sorted(stats)

    fig, ax = plt.subplots(figsize=(9, 6), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, color=GRIDLINE, linewidth=1, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED)
    ax.set_ylim(-5, 108)
    ax.axhline(50, color=INK_MUTED, linewidth=1, linestyle="--", zorder=1, label="_chance")

    if args.openrouter_cap in num_frames_values or (
        num_frames_values and min(num_frames_values) < args.openrouter_cap < max(num_frames_values)
    ):
        ax.axvline(
            args.openrouter_cap, color=CAP_LINE_COLOR, linewidth=1.5, linestyle=":",
            zorder=1, label=f"OpenRouter cap ({args.openrouter_cap} frames)",
        )

    ys, errs, n_per_point = [], [], 0
    for nf in num_frames_values:
        acc, sem, n = stats[nf]
        ys.append(acc)
        errs.append(sem)
        n_per_point = max(n_per_point, n)
    ax.errorbar(
        num_frames_values, ys, yerr=errs, marker="o", markersize=7, linewidth=2,
        color=LINE_COLOR, ecolor=LINE_COLOR, elinewidth=1.5, capsize=4,
        label="gemma-4-31b-it (local)", zorder=3,
    )

    ax.set_xticks(num_frames_values)
    ax.set_xlabel("num_frames (sampled from video)", color=INK_SECONDARY)
    ax.set_ylabel("Accuracy (%)", color=INK_SECONDARY)

    with open(results_csv) as f:
        first_row = next(csv.DictReader(f))
    n_objects = first_row["n_objects"]
    rotation_deg = first_row["rotation_deg"]
    config_path = f"results/debug_circular/{args.run_name}/config.json"
    with open(config_path) as f:
        model = json.load(f)["model"]

    fig.suptitle(
        f"{model} on debug_circular (n_objects={n_objects}, rotation_deg={rotation_deg}): "
        "accuracy vs. num_frames sampled\n"
        f"(error bars: SEM, n={n_per_point}/point; dashed line: chance; "
        "dotted line: OpenRouter's measured 32-frame cap)",
        color=INK_PRIMARY, fontsize=12.5, y=1.0,
    )
    ax.legend(frameon=False, labelcolor=INK_PRIMARY, fontsize=10, loc="lower left")

    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
