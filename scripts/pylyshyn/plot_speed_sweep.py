"""
Summary figure for the pylyshyn redirect-cadence x speed x native/stretched
sweep on qwen3.6-plus (results/pylyshyn/speed_sweep/results.csv, produced by
run_pylyshyn_speed_sweep.py -- see claude/2026_08/2026_08_27/TODO.md's
"Task 2"): a 2x2 grid of panels, rows = redirect_condition (slow=2.0s,
fast=1.0s), columns = speed_condition (slow=16-33px/s, fast=32-66px/s), each
panel showing the native/stretched accuracy bar pair with SEM -- same
colors/style/bar-pair convention as plot_stretch_sweep.py.

Usage:
    uv run python scripts/pylyshyn/plot_speed_sweep.py
"""

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

RESULTS_CSV = "results/pylyshyn/speed_sweep/results.csv"
OUT_PATH = "results/pylyshyn/speed_sweep/accuracy_summary.png"

REDIRECT_ORDER = ["slow", "fast"]
SPEED_ORDER = ["slow", "fast"]
REDIRECT_LABELS = {"slow": "redirect_s = 2.0", "fast": "redirect_s = 1.0"}
SPEED_LABELS = {"slow": "speed_px_s = 16-33", "fast": "speed_px_s = 32-66"}
VARIANT_LABELS = {"native": "Native (fps=10, ~10s)", "stretched": "Stretched (fps=2, ~50s)"}


def _parse_bool(s: str) -> bool | None:
    if s == "True":
        return True
    if s == "False":
        return False
    return None


def accuracy_by_group(results_csv: str) -> dict[tuple[str, str, str], tuple[float, float, int]]:
    """Returns {(redirect_condition, speed_condition, variant): (accuracy_pct, sem_pct, n)}."""
    totals = defaultdict(int)
    correct = defaultdict(int)
    with open(results_csv) as f:
        for row in csv.DictReader(f):
            key = (row["redirect_condition"], row["speed_condition"], row["variant"])
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
    stats = accuracy_by_group(RESULTS_CSV)

    fig, axes = plt.subplots(
        len(REDIRECT_ORDER), len(SPEED_ORDER), figsize=(11, 8),
        facecolor="#fcfcfb", sharey=True, sharex=True,
    )
    fig.suptitle(
        "pylyshyn on qwen3.6-plus (no reasoning): redirect cadence x motion speed\n"
        "(error bars: SEM, n=16/bar; dashed line: chance)",
        color=INK_PRIMARY, fontsize=12.5, y=0.99,
    )

    bar_width = 0.4
    xs = [0, 1]
    n_seen = 0
    for row_i, redirect_condition in enumerate(REDIRECT_ORDER):
        for col_i, speed_condition in enumerate(SPEED_ORDER):
            ax = axes[row_i][col_i]
            ax.set_facecolor("#fcfcfb")
            ax.grid(True, axis="y", color=GRIDLINE, linewidth=1, zorder=0)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            for spine in ("left", "bottom"):
                ax.spines[spine].set_color(BASELINE)
            ax.tick_params(colors=INK_MUTED)
            ax.set_ylim(0, 108)
            ax.axhline(50, color=INK_MUTED, linewidth=1, linestyle="--", zorder=1)

            for offset, variant in zip((-1, 1), ("native", "stretched")):
                acc, sem, n = stats.get((redirect_condition, speed_condition, variant), (0, 0, 0))
                n_seen += n
                x = xs[0] + offset * bar_width / 2 + 0.5
                ax.bar(
                    [x], [acc], width=bar_width, color=VARIANT_COLORS[variant],
                    label=VARIANT_LABELS[variant], zorder=3, edgecolor="#fcfcfb", linewidth=2,
                )
                ax.errorbar(
                    [x], [acc], yerr=[sem], fmt="none",
                    ecolor=INK_PRIMARY, elinewidth=1.5, capsize=4, zorder=4,
                )

            ax.set_xlim(0, 1)
            ax.set_xticks([])
            ax.set_title(
                f"{REDIRECT_LABELS[redirect_condition]}, {SPEED_LABELS[speed_condition]}",
                color=INK_PRIMARY, fontsize=11,
            )
            if col_i == 0:
                ax.set_ylabel("Accuracy (%)", color=INK_SECONDARY)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles, labels, frameon=False, labelcolor=INK_PRIMARY, fontsize=10,
        loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.90),
    )

    if n_seen == 0:
        print(f"Warning: no rows found in {RESULTS_CSV}")

    fig.subplots_adjust(top=0.78, hspace=0.35, wspace=0.12)
    fig.savefig(OUT_PATH, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
