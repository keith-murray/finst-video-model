"""
Summary figure for the pylyshyn native-vs-stretched-duration x reasoning
sweep (results/pylyshyn/stretch_sweep/results.csv, produced by
run_pylyshyn_stretch_sweep.py -- see claude/2026_08/2026_08_26/TODO.md's
"Part 3" and claude/2026_08/2026_08_27/TODO.md's "Task 1").

Single-panel design (2026-08-27 rewrite of the original one-subplot-per-model
layout): every (model, reasoning_level) condition gets one x-tick (model on
the first label line, level on the second), with the native/stretched bar
pair + SEM at each tick, all sharing one axis. The old per-model-subplot
layout gave every panel the same *physical* width regardless of how many
levels that model had, so a model with only one level (e.g. qwen3.6-plus)
got comically wide bars -- a single shared axis makes every bar the same
width regardless of how many conditions a given model has.

Note baked into the footnote: qwen3.8-27b's "low"/"medium" reasoning_level
conditions run at a small, reproducible internal reasoning-token budget
(~65/~257 tokens) that a max_tokens fix (both top-level and the
mutually-exclusive-with-effort reasoning.max_tokens field) could not raise
-- investigated 2026-08-26, see finst_video_model.vlm_client
.ask_about_video's docstring for the full writeup. This is accepted as a
real property of this reasoning control on this model/endpoint, not a
truncation bug -- the data below is treated as trustworthy at face value.
"high" (added 2026-08-27) is not subject to this pinning -- see
[[reference-openrouter-reasoning-max-tokens]].

Mirrors scripts/debug_circular/plot_openrouter_diagnostic.py's design
exactly (same validated categorical slots 1-2: blue=native, orange=stretched).

Usage:
    uv run python scripts/pylyshyn/plot_stretch_sweep.py
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
FOOTNOTE = "#6b6a63"
VARIANT_COLORS = {"native": "#2a78d6", "stretched": "#eb6834"}  # validated categorical slots 1-2

RESULTS_CSV = "results/pylyshyn/stretch_sweep/results.csv"
OUT_PATH = "results/pylyshyn/stretch_sweep/accuracy_summary.png"

REASONING_LEVEL_ORDER = ["none", "minimal", "low", "medium", "high"]
MODEL_LABELS = {
    "qwen/qwen3.8-27b": "qwen3.8-27b",
    "qwen/qwen3.8-max": "qwen3.8-max",
    "qwen/qwen3.5-122b-a10b": "qwen3.5-122b-a10b\n(10B active)",
    "qwen/qwen3.6-plus": "qwen3.6-plus",
}
MODEL_ORDER = [
    "qwen/qwen3.8-27b", "qwen/qwen3.8-max",
    "qwen/qwen3.5-122b-a10b", "qwen/qwen3.6-plus",
]
VARIANT_LABELS = {"native": "Native (fps=10, ~10s)", "stretched": "Stretched (fps=2, ~50s)"}

# (model, reasoning_level) pairs running at a small, reproducible reasoning
# budget (~65/~257 tokens, see module docstring) -- marked on the chart for
# context, not excluded from it.
SMALL_BUDGET_CONDITIONS = {("qwen/qwen3.8-27b", "low"), ("qwen/qwen3.8-27b", "medium")}


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
            key = (row["model"], row["variant"], row["reasoning_level"])
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

    # Flat list of (model, level) conditions, grouped by model, in MODEL_ORDER/
    # REASONING_LEVEL_ORDER order -- this is what gives every model's bars the
    # same width, unlike the old one-subplot-per-model layout.
    conditions = [
        (model, level)
        for model in MODEL_ORDER
        for level in REASONING_LEVEL_ORDER
        if (model, "native", level) in stats
    ]

    fig, ax = plt.subplots(figsize=(max(9.0, 1.7 * len(conditions)), 6.0), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, axis="y", color=GRIDLINE, linewidth=1, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED)
    ax.set_ylim(0, 108)
    ax.axhline(50, color=INK_MUTED, linewidth=1, linestyle="--", zorder=1)

    bar_width = 0.32
    xs = range(len(conditions))
    for offset, variant in zip((-1, 1), ("native", "stretched")):
        heights, errs = [], []
        for model, level in conditions:
            acc, sem, n = stats.get((model, variant, level), (0, 0, 0))
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

    # Thin dividers between model groups.
    for i in range(1, len(conditions)):
        if conditions[i][0] != conditions[i - 1][0]:
            ax.axvline(i - 0.5, color=BASELINE, linewidth=1, zorder=2)

    xtick_labels = [
        f"{MODEL_LABELS[model]}\n{level}{'*' if (model, level) in SMALL_BUDGET_CONDITIONS else ''}"
        for model, level in conditions
    ]
    ax.set_xticks(list(xs))
    ax.set_xticklabels(xtick_labels)
    ax.set_xlim(-0.6, len(conditions) - 0.4)
    ax.set_ylabel("Accuracy (%)", color=INK_SECONDARY)

    fig.tight_layout(rect=(0, 0, 1, 0.86))

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels, frameon=False, labelcolor=INK_PRIMARY, fontsize=10,
        loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.94),
    )
    fig.suptitle(
        "pylyshyn via OpenRouter: native vs. stretched-duration mp4 encoding\n"
        "(error bars: SEM, n=16/bar; dashed line: chance)",
        color=INK_PRIMARY, fontsize=12, y=1.0,
    )
    fig.text(
        0.5, 0.895,
        "* qwen3.8-27b low/medium: reasoning runs at a small, reproducible internal\n"
        "budget (~65/~257 tokens) that max_tokens does not raise -- see script docstring",
        color=FOOTNOTE, fontsize=9.5, ha="center", va="top", style="italic",
    )

    fig.savefig(OUT_PATH, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
