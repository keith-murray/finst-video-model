"""
Summary figure for claude/2026_08/2026_08_27/TODO.md's "Task 3":
redirect-cadence x speed x native/stretched sweep across models
(results/pylyshyn/speed_sweep_models/results.csv, produced by
run_pylyshyn_speed_sweep_models.py). One row of 4 panels (redirect x speed
conditions, in the same order as plot_speed_sweep.py) per model, each panel
showing the native/stretched accuracy bar pair with SEM -- same colors/
style as plot_speed_sweep.py, generalized to stack additional model rows
if/when more models are added to MODEL_REASONING.

Usage:
    uv run python scripts/pylyshyn/plot_speed_sweep_models.py
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

RESULTS_CSV = "results/pylyshyn/speed_sweep_models/results.csv"
OUT_PATH = "results/pylyshyn/speed_sweep_models/accuracy_summary.png"

MODEL_ORDER = ["z-ai/glm-5v-turbo"]
REDIRECT_ORDER = ["slow", "fast"]
SPEED_ORDER = ["slow", "fast"]
REDIRECT_LABELS = {"slow": "redirect_s=2.0", "fast": "redirect_s=1.0"}
SPEED_LABELS = {"slow": "speed_px_s=16-33", "fast": "speed_px_s=32-66"}
VARIANT_LABELS = {"native": "Native (fps=10, ~10s)", "stretched": "Stretched (fps=2, ~50s)"}


def _parse_bool(s: str) -> bool | None:
    if s == "True":
        return True
    if s == "False":
        return False
    return None


def accuracy_by_group(results_csv: str) -> dict[tuple[str, str, str, str], tuple[float, float, int]]:
    """Returns {(model, redirect_condition, speed_condition, variant): (accuracy_pct, sem_pct, n)}."""
    totals = defaultdict(int)
    correct = defaultdict(int)
    with open(results_csv) as f:
        for row in csv.DictReader(f):
            key = (row["model"], row["redirect_condition"], row["speed_condition"], row["variant"])
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
    models_present = [m for m in MODEL_ORDER if any(k[0] == m for k in stats)]
    conditions = [(r, s) for r in REDIRECT_ORDER for s in SPEED_ORDER]

    # Fixed ~2in header (suptitle + legend) regardless of row count, so the
    # header doesn't collide with panel titles when there's only one model row.
    header_in = 2.0
    row_in = 4.2
    fig_h = header_in + row_in * len(models_present)
    fig, axes = plt.subplots(
        len(models_present), len(conditions), figsize=(4.2 * len(conditions), fig_h),
        facecolor="#fcfcfb", sharey=True, squeeze=False,
    )
    fig.suptitle(
        "pylyshyn: redirect cadence x motion speed across models\n"
        "(error bars: SEM, n/bar shown per panel; dashed line: chance)",
        color=INK_PRIMARY, fontsize=12.5, y=0.995,
    )

    for row_i, model in enumerate(models_present):
        for col_i, (redirect_condition, speed_condition) in enumerate(conditions):
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

            bar_width = 0.4
            n_this_panel = 0
            for offset, variant in zip((-1, 1), ("native", "stretched")):
                acc, sem, n = stats.get((model, redirect_condition, speed_condition, variant), (0, 0, 0))
                n_this_panel = max(n_this_panel, n)
                x = 0.5 + offset * bar_width / 2
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
            title = f"{REDIRECT_LABELS[redirect_condition]}, {SPEED_LABELS[speed_condition]}"
            if row_i == 0:
                ax.set_title(f"{title}\n(n={n_this_panel}/bar)", color=INK_PRIMARY, fontsize=10.5)
            else:
                ax.set_title(f"(n={n_this_panel}/bar)", color=INK_PRIMARY, fontsize=10.5)
            if col_i == 0:
                ax.set_ylabel(f"{model}\nAccuracy (%)", color=INK_SECONDARY, fontsize=10)

    # bbox_to_anchor/top are figure-fraction, so a fixed header_in maps to a
    # different fraction depending on fig_h -- compute both from header_in.
    legend_y = 1 - 0.85 / fig_h
    top = 1 - header_in / fig_h

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles, labels, frameon=False, labelcolor=INK_PRIMARY, fontsize=10,
        loc="upper center", ncol=2, bbox_to_anchor=(0.5, legend_y),
    )

    fig.subplots_adjust(top=top, hspace=0.5, wspace=0.12)
    fig.savefig(OUT_PATH, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
