"""
Bar chart for `claude/2026_09/2026_09_16/TODO.md`'s Part 4 (no-reasoning
pass): compares accuracy across 4 OpenRouter models on the 100-trial,
n_objects=3 pylyshyn set deliberately constructed so the nearest-neighbor
position heuristic is **wrong on every trial** (`generate_hardest_stimuli.py`,
0.0% by construction) -- an even harder follow-up to Part 1's
heuristic-neutralized-to-chance set. `google/gemini-3.8-flash` is excluded
here (mandatory reasoning on OpenRouter, can't run reasoning-off -- see
`run_hardest_stimuli_sweep.py`'s docstring); it appears only in
`plot_hardest_stimuli_thinking_sweep.py`.

Error bars are SEM, bar-plus-errorbar structure mirrors every other
`plot_*.py` in this directory.

Reads `aggregate_hardest_stimuli_sweep.py`'s summary.json rather than
recomputing from results.csv, so the two stay in sync.

Usage:
    uv run python scripts/pylyshyn/hard_stimuli/plot_hardest_stimuli_sweep.py
"""

import json
import os

import matplotlib.pyplot as plt

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

MODEL_COLORS = {
    "google/gemma-4-31b-it": "#eb6834",
    "qwen/qwen3.6-plus": "#8a5fc2",
    "moonshotai/kimi-k3": "#2fa85a",
    "qwen/qwen3.8-27b": "#c2a02a",
    "google/gemini-3.8-flash": "#2a78d6",
}
MODEL_LABELS = {
    "google/gemma-4-31b-it": "gemma-4-31b-it",
    "qwen/qwen3.6-plus": "qwen3.6-plus",
    "moonshotai/kimi-k3": "kimi-k3",
    "qwen/qwen3.8-27b": "qwen3.8-27b",
    "google/gemini-3.8-flash": "gemini-3.8-flash",
}
DEFAULT_COLOR = "#8a897f"

SUMMARY_PATH = "results/pylyshyn/hard_stimuli/hardest_stimuli_sweep/summary.json"
OUT_PATH = "results/pylyshyn/hard_stimuli/hardest_stimuli_sweep/accuracy_by_model.png"


def main():
    with open(SUMMARY_PATH) as f:
        summary = json.load(f)

    models = [k for k in summary if not k.startswith("_")]
    heuristic = summary["_heuristic"]

    fig, ax = plt.subplots(figsize=(9.5, 6), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, axis="y", color=GRIDLINE, linewidth=1, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED)
    ax.set_ylim(0, 118)
    ax.axhline(
        heuristic["accuracy_pct"], color=INK_MUTED, linewidth=1, linestyle="--", zorder=1,
        label=f"Nearest-neighbor heuristic ({heuristic['accuracy_pct']:.0f}%, always wrong by construction)",
    )

    xs = range(len(models))
    heights = [summary[m]["accuracy_pct"] for m in models]
    sems = [summary[m]["accuracy_sem_pct"] for m in models]
    colors = [MODEL_COLORS.get(m, DEFAULT_COLOR) for m in models]
    ax.bar(xs, heights, width=0.55, color=colors, zorder=3, edgecolor="#fcfcfb", linewidth=2)
    ax.errorbar(
        xs, heights, yerr=sems, fmt="none", ecolor=INK_PRIMARY, elinewidth=1.5, capsize=5, zorder=4,
    )

    ax.set_xticks(list(xs))
    ax.set_xticklabels([
        f"{MODEL_LABELS.get(m, m)}\n(n={summary[m]['n_trials']})" for m in models
    ])
    ax.set_xlim(-0.6, len(models) - 0.4)
    ax.set_ylabel("Accuracy (%)", color=INK_SECONDARY)
    ax.legend(loc="upper right", frameon=False, labelcolor=INK_SECONDARY, fontsize=9)

    fig.suptitle(
        "pylyshyn n_objects=3, stretched, no reasoning:\n"
        "accuracy across models on the heuristic-always-wrong 100-trial set\n"
        "(error bars: SEM; dashed line: nearest-neighbor heuristic, always wrong)",
        color=INK_PRIMARY, fontsize=11, y=1.0,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(OUT_PATH, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print(f"Wrote {OUT_PATH}")
    for m in models:
        s = summary[m]
        print(
            f"  {MODEL_LABELS.get(m, m)}: {s['accuracy_pct']:.1f}% (sem={s['accuracy_sem_pct']:.1f}%, "
            f"n={s['n_trials']}), d'={s['d_prime']:.2f}, cost=${s['total_cost_usd']:.4f}"
        )
    print(f"  heuristic: {heuristic['accuracy_pct']:.1f}% (sem={heuristic['accuracy_sem_pct']:.1f}%, n={heuristic['n_trials']})")


if __name__ == "__main__":
    main()
