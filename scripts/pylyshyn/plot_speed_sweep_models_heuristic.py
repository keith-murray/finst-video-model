"""
Follow-up figure for `claude/2026_09/2026_09_01/TODO.md`'s Task 2: the
V-shaped debug_circular angle curve ([[project-status-2026-09-01]]) raised
the concern that gemma-4-31b-it/qwen3.6-plus's strong pylyshyn scores
(`plot_speed_sweep_models.py`'s `accuracy_summary.png`) might reflect a
"nearest neighbor" position heuristic (see
`scripts/pylyshyn/nearest_neighbor_heuristic.py`) rather than genuine
identity tracking through the motion.

Re-draws just the qwen3.6-plus and gemma-4-31b-it rows of that same 4-panel
(redirect x speed) x (native/stretched bars) layout, adding a dashed red
horizontal line per panel for the heuristic's accuracy on that exact same
set of trials (same underlying stimuli, re-simulated from each trial's
saved `ground_truth.json` -- no video/API calls, see the heuristic module's
docstring for why this is model/variant-agnostic). Written to a new file
(`accuracy_summary_heuristic.png`) rather than overwriting the original
5-model `accuracy_summary.png`, so that committed comparison across all 5
models is preserved.

Usage:
    uv run python scripts/pylyshyn/plot_speed_sweep_models_heuristic.py
"""

import csv

import matplotlib.pyplot as plt

from nearest_neighbor_heuristic import heuristic_accuracy_by_group
from plot_speed_sweep_models import (
    INK_MUTED, INK_PRIMARY, INK_SECONDARY, BASELINE, GRIDLINE,
    REDIRECT_LABELS, SPEED_LABELS, VARIANT_COLORS, VARIANT_LABELS,
    accuracy_by_group,
)

MODELS = [
    ("qwen/qwen3.6-plus", "qwen3.6-plus", "results/pylyshyn/speed_sweep/results.csv", "data/pylyshyn/speed_sweep/trials"),
    ("google/gemma-4-31b-it", "gemma-4-31b-it", "results/pylyshyn/speed_sweep_models/results.csv", "data/pylyshyn/speed_sweep_models/trials"),
]
OUT_PATH = "results/pylyshyn/speed_sweep_models/accuracy_summary_heuristic.png"

REDIRECT_ORDER = ["slow", "fast"]
SPEED_ORDER = ["slow", "fast"]
HEURISTIC_COLOR = "#d1332e"  # distinct red, doesn't collide with VARIANT_COLORS


def main():
    conditions = [(r, s) for r in REDIRECT_ORDER for s in SPEED_ORDER]

    per_model_stats = {}
    per_model_heuristic = {}
    for model, _, results_csv, trials_dir in MODELS:
        with open(results_csv) as f:
            rows = [r for r in csv.DictReader(f) if r["model"] == model]
        per_model_stats[model] = accuracy_by_group([results_csv])
        per_model_heuristic[model] = heuristic_accuracy_by_group(
            rows, trials_dir, lambda row: (row["redirect_condition"], row["speed_condition"]),
        )

    header_in = 2.0
    row_in = 4.2
    fig_h = header_in + row_in * len(MODELS)
    fig, axes = plt.subplots(
        len(MODELS), len(conditions), figsize=(4.2 * len(conditions), fig_h),
        facecolor="#fcfcfb", sharey=True, squeeze=False,
    )
    fig.suptitle(
        "pylyshyn: redirect cadence x motion speed, model vs. nearest-neighbor heuristic\n"
        "(error bars: SEM, n/bar shown per panel; gray dashed: chance; red dashed: heuristic)",
        color=INK_PRIMARY, fontsize=12.5, y=0.995,
    )

    for row_i, (model, model_label, _, _) in enumerate(MODELS):
        stats = per_model_stats[model]
        heuristic_stats = per_model_heuristic[model]
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
            ax.axhline(50, color=INK_MUTED, linewidth=1, linestyle="--", zorder=1, label="_chance")

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

            h_acc, h_sem, h_n = heuristic_stats.get((redirect_condition, speed_condition), (float("nan"), 0, 0))
            ax.axhline(
                h_acc, color=HEURISTIC_COLOR, linewidth=1.5, linestyle="--", zorder=2,
                label="_heuristic" if not (row_i == 0 and col_i == 0) else "Nearest-neighbor heuristic",
            )

            ax.set_xlim(0, 1)
            ax.set_xticks([])
            title = f"{REDIRECT_LABELS[redirect_condition]}, {SPEED_LABELS[speed_condition]}"
            subtitle = f"(n={n_this_panel}/bar, heuristic n={h_n})"
            if row_i == 0:
                ax.set_title(f"{title}\n{subtitle}", color=INK_PRIMARY, fontsize=10.5)
            else:
                ax.set_title(subtitle, color=INK_PRIMARY, fontsize=10.5)
            if col_i == 0:
                ax.set_ylabel(f"{model_label}\nAccuracy (%)", color=INK_SECONDARY, fontsize=10)

    legend_y = 1 - 0.85 / fig_h
    top = 1 - header_in / fig_h

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles, labels, frameon=False, labelcolor=INK_PRIMARY, fontsize=10,
        loc="upper center", ncol=3, bbox_to_anchor=(0.5, legend_y),
    )

    fig.subplots_adjust(top=top, hspace=0.5, wspace=0.12)
    fig.savefig(OUT_PATH, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
