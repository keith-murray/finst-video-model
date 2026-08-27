"""
Summary figure for claude/2026_08/2026_08_27/TODO.md's "Task 4":
google/gemma-4-31b-it on debug_circular across a rotation_deg sweep, native
vs. stretched (results/debug_circular/angle_sweep/results.csv, produced by
run_debug_circular_angle_sweep.py). Line plot: rotation_deg on the x-axis,
accuracy (%) on the y-axis, one line per variant (native/stretched) with
SEM error bars -- per the user's explicit request for a line plot with two
series, rather than this project's usual grouped-bar convention.

Per [[reference-gemma-fixed-frame-budget]], the two lines are expected to
closely track each other (gemma-4-31b-it's frame sampling is insensitive to
both real and declared video duration) -- this is a sanity-check
confirmation across a different (interpolable-rotation, not pylyshyn's
random-walk) motion type, not expected to reveal a stretch effect. (It
also surfaced an unpredicted V-shaped, below-chance-in-the-middle accuracy
curve -- see [[project-status-2026-08-27]]'s Task 4.)

`--run-name` selects which `results/debug_circular/<run-name>/results.csv`
to plot (e.g. the `angle_sweep_n4` position-heuristic follow-up, which is
native-only -- the plot only draws whichever variants are actually present).

Usage:
    uv run python scripts/debug_circular/plot_angle_sweep.py
    uv run python scripts/debug_circular/plot_angle_sweep.py --run-name angle_sweep_n4
"""

import argparse
import csv
import math
from collections import defaultdict

import matplotlib.pyplot as plt

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
VARIANT_COLORS = {"native": "#2a78d6", "stretched": "#eb6834"}  # validated categorical slots 1-2
VARIANT_LABELS = {"native": "Native (fps=10, ~10s)", "stretched": "Stretched (fps=2, ~50s)"}


def _parse_bool(s: str) -> bool | None:
    if s == "True":
        return True
    if s == "False":
        return False
    return None


def accuracy_by_group(results_csv: str) -> dict[tuple[float, str], tuple[float, float, int]]:
    """Returns {(rotation_deg, variant): (accuracy_pct, sem_pct, n)}."""
    totals = defaultdict(int)
    correct = defaultdict(int)
    with open(results_csv) as f:
        for row in csv.DictReader(f):
            key = (float(row["rotation_deg"]), row["variant"])
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
    parser.add_argument("--run-name", type=str, default="angle_sweep")
    args = parser.parse_args()

    results_csv = f"results/debug_circular/{args.run_name}/results.csv"
    out_path = f"results/debug_circular/{args.run_name}/accuracy_by_angle.png"

    stats = accuracy_by_group(results_csv)
    rotation_degs = sorted({k[0] for k in stats})
    variants_present = [v for v in ("native", "stretched") if any(k[1] == v for k in stats)]

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

    n_per_point = 0
    for variant in variants_present:
        ys, errs = [], []
        for rd in rotation_degs:
            acc, sem, n = stats.get((rd, variant), (float("nan"), 0, 0))
            ys.append(acc)
            errs.append(sem)
            n_per_point = max(n_per_point, n)
        ax.errorbar(
            rotation_degs, ys, yerr=errs, marker="o", markersize=7, linewidth=2,
            color=VARIANT_COLORS[variant], ecolor=VARIANT_COLORS[variant],
            elinewidth=1.5, capsize=4, label=VARIANT_LABELS[variant], zorder=3,
        )

    ax.set_xticks(rotation_degs)
    ax.set_xlabel("rotation_deg", color=INK_SECONDARY)
    ax.set_ylabel("Accuracy (%)", color=INK_SECONDARY)

    with open(results_csv) as f:
        n_objects = next(csv.DictReader(f))["n_objects"]

    fig.suptitle(
        f"google/gemma-4-31b-it on debug_circular (n_objects={n_objects}): "
        "accuracy vs. rotation angle\n"
        f"(error bars: SEM, n={n_per_point}/point; dashed line: chance)",
        color=INK_PRIMARY, fontsize=12.5, y=1.0,
    )
    ax.legend(frameon=False, labelcolor=INK_PRIMARY, fontsize=10, loc="lower left")

    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
