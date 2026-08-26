"""
Summary figure for the OpenRouter native-vs-stretched-duration diagnostic
(results/debug_circular/openrouter_diagnostic/results.csv, produced by
run_openrouter_diagnostic.py -- see claude/2026_08/2026_08_26/TODO.md's
"Part 2"): accuracy (%) on the y-axis, reasoning effort level on the
x-axis, one subplot per model, two bars per x-position (native vs.
stretched video encoding) with SEM error bars.

An unparseable answer counts as incorrect (denominator is every trial, not
just parseable ones), matching scripts/pylyshyn/plot_reasoning_summary.py's
percent-error convention. qwen3.8-27b has a "none" (reasoning-disabled)
point that qwen3.8-max lacks (reasoning is mandatory there), so its panel
has one more x-tick.

Colors follow scripts/pylyshyn/plot_qwen_summary.py's convention (validated
categorical slots 1-2: blue, orange) -- here color encodes video variant
(native/stretched), consistent across both model subplots, rather than
model (which plot_reasoning_summary.py used blue/orange for instead) --
model is the facet dimension in this figure, variant is the series
dimension.

Usage:
    uv run python scripts/debug_circular/plot_openrouter_diagnostic.py
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

RESULTS_CSV = "results/debug_circular/openrouter_diagnostic/results.csv"
OUT_PATH = "results/debug_circular/openrouter_diagnostic/accuracy_summary.png"

REASONING_LEVEL_ORDER = ["none", "minimal", "low", "medium", "high"]
MODEL_LABELS = {
    "qwen/qwen3.8-27b": "qwen3.8-27b",
    "qwen/qwen3.8-max": "qwen3.8-max",
}
MODEL_ORDER = ["qwen/qwen3.8-27b", "qwen/qwen3.8-max"]
VARIANT_LABELS = {"native": "Native (fps=10, ~10s)", "stretched": "Stretched (fps=2, ~50s)"}


def _parse_bool(s: str) -> bool | None:
    if s == "True":
        return True
    if s == "False":
        return False
    return None


def accuracy_by_group(results_csv: str) -> dict[tuple[str, str, str], tuple[float, float, int]]:
    """Returns {(model, variant, reasoning_level): (accuracy_pct, sem_pct, n)}."""
    totals = defaultdict(int)
    correct = defaultdict(int)
    with open(results_csv) as f:
        for row in csv.DictReader(f):
            key = (row["model"], row["video_variant"], row["reasoning_level"])
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

    fig, axes = plt.subplots(
        1, len(models_present), figsize=(6.5 * len(models_present), 5.5),
        facecolor="#fcfcfb", sharey=True,
    )
    if len(models_present) == 1:
        axes = [axes]

    bar_width = 0.32
    for ax, model in zip(axes, models_present):
        ax.set_facecolor("#fcfcfb")
        ax.grid(True, axis="y", color=GRIDLINE, linewidth=1, zorder=0)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(BASELINE)
        ax.tick_params(colors=INK_MUTED)
        ax.set_ylim(0, 108)
        ax.axhline(50, color=INK_MUTED, linewidth=1, linestyle="--", zorder=1)

        levels = [lvl for lvl in REASONING_LEVEL_ORDER if (model, "native", lvl) in stats]
        xs = range(len(levels))

        for offset, variant in zip((-1, 1), ("native", "stretched")):
            heights, errs = [], []
            for lvl in levels:
                acc, sem, n = stats.get((model, variant, lvl), (0, 0, 0))
                heights.append(acc)
                errs.append(sem)
            ax.bar(
                [x + offset * bar_width / 2 for x in xs], heights, width=bar_width,
                color=VARIANT_COLORS[variant], label=VARIANT_LABELS[variant],
                zorder=3, edgecolor="#fcfcfb", linewidth=2,
            )
            ax.errorbar(
                [x + offset * bar_width / 2 for x in xs], heights, yerr=errs,
                fmt="none", ecolor=INK_PRIMARY, elinewidth=1.5, capsize=4, zorder=4,
            )

        ax.set_xticks(list(xs))
        ax.set_xticklabels(levels)
        ax.set_xlim(-0.6, len(levels) - 0.4)
        ax.set_xlabel("Reasoning effort", color=INK_SECONDARY)
        ax.set_title(MODEL_LABELS[model], color=INK_PRIMARY, fontsize=13, fontweight="bold")

    axes[0].set_ylabel("Accuracy (%)", color=INK_SECONDARY)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, frameon=False, labelcolor=INK_PRIMARY, fontsize=10,
        loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.04),
    )
    fig.suptitle(
        "debug_circular via OpenRouter: native vs. stretched-duration mp4 encoding\n"
        "(error bars: SEM, n=16/bar; dashed line: chance)",
        color=INK_PRIMARY, fontsize=12, y=1.14,
    )

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
