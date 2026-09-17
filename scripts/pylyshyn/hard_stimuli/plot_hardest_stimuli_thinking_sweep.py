"""
Grouped bar chart for `claude/2026_09/2026_09_16/TODO.md`'s Part 4
(reasoning off vs. "medium", even-harder heuristic-always-wrong stimulus
set). Compares `run_hardest_stimuli_sweep.py` (no reasoning, 4 models) vs.
`run_hardest_stimuli_thinking_sweep.py` (reasoning="medium", 5 models)
accuracy per model, side by side, on the identical 100-trial
heuristic-always-wrong pylyshyn n_objects=3 set.

`google/gemini-3.8-flash` has no no-reasoning bar (mandatory reasoning on
OpenRouter, excluded from that pass entirely -- see
`run_hardest_stimuli_sweep.py`'s docstring) -- drawn as a reasoning-only
bar instead of being dropped from the figure, since its result (100%
accuracy, d'=4.67) is the single most notable finding in this pass.

Reads both `aggregate_hardest_stimuli_sweep.py`'s (no-reasoning) and
`aggregate_hardest_stimuli_thinking_sweep.py`'s (medium-reasoning)
summary.json files.

Usage:
    uv run python scripts/pylyshyn/hard_stimuli/plot_hardest_stimuli_thinking_sweep.py
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

NO_REASONING_COLOR = "#8a897f"
REASONING_COLOR = "#2a78d6"

MODEL_LABELS = {
    "google/gemma-4-31b-it": "gemma-4-31b-it",
    "qwen/qwen3.6-plus": "qwen3.6-plus",
    "moonshotai/kimi-k3": "kimi-k3",
    "qwen/qwen3.8-27b": "qwen3.8-27b",
    "google/gemini-3.8-flash": "gemini-3.8-flash\n(reasoning mandatory)",
}

NO_REASONING_SUMMARY = "results/pylyshyn/hard_stimuli/hardest_stimuli_sweep/summary.json"
REASONING_SUMMARY = "results/pylyshyn/hard_stimuli/hardest_stimuli_thinking_sweep/summary.json"
OUT_PATH = "results/pylyshyn/hard_stimuli/hardest_stimuli_thinking_sweep/accuracy_no_reasoning_vs_medium_reasoning.png"


def main():
    with open(NO_REASONING_SUMMARY) as f:
        no_reasoning = json.load(f)
    with open(REASONING_SUMMARY) as f:
        reasoning = json.load(f)

    # Union of models across both passes (not just the intersection), so a
    # model with no no-reasoning bar (gemini-3.8-flash) still appears.
    models = [m for m in MODEL_LABELS if m in no_reasoning or m in reasoning]
    heuristic = reasoning["_heuristic"]

    fig, ax = plt.subplots(figsize=(10.5, 6.2), facecolor="#fcfcfb")
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
    ax.axhline(80, color="#c9a227", linewidth=1, linestyle=":", zorder=1, label="80% (TODO's benchmark)")

    xs = np.arange(len(models))
    width = 0.36
    no_r_heights = [no_reasoning[m]["accuracy_pct"] if m in no_reasoning else 0.0 for m in models]
    no_r_sems = [no_reasoning[m]["accuracy_sem_pct"] if m in no_reasoning else 0.0 for m in models]
    r_heights = [reasoning[m]["accuracy_pct"] for m in models]
    r_sems = [reasoning[m]["accuracy_sem_pct"] for m in models]

    ax.bar(
        xs - width / 2, no_r_heights, width=width, color=NO_REASONING_COLOR, zorder=3,
        edgecolor="#fcfcfb", linewidth=1.5, label="Reasoning off",
    )
    ax.errorbar(xs - width / 2, no_r_heights, yerr=no_r_sems, fmt="none", ecolor=INK_PRIMARY, elinewidth=1.3, capsize=4, zorder=4)
    ax.bar(
        xs + width / 2, r_heights, width=width, color=REASONING_COLOR, zorder=3,
        edgecolor="#fcfcfb", linewidth=1.5, label="Reasoning = medium",
    )
    ax.errorbar(xs + width / 2, r_heights, yerr=r_sems, fmt="none", ecolor=INK_PRIMARY, elinewidth=1.3, capsize=4, zorder=4)

    ax.set_xticks(list(xs))
    ax.set_xticklabels([
        f"{MODEL_LABELS[m]}\n(n={no_reasoning[m]['n_trials'] if m in no_reasoning else '--'}/{reasoning[m]['n_trials']})"
        for m in models
    ])
    ax.set_xlim(-0.7, len(models) - 0.3)
    ax.set_ylabel("Accuracy (%)", color=INK_SECONDARY)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False, labelcolor=INK_SECONDARY, fontsize=9)

    fig.suptitle(
        "pylyshyn n_objects=3, stretched, heuristic-always-wrong:\n"
        "reasoning off vs. reasoning=\"medium\", accuracy by model\n"
        "(error bars: SEM; gemini-3.8-flash has mandatory reasoning, no \"off\" bar)",
        color=INK_PRIMARY, fontsize=11, y=1.0,
    )

    fig.tight_layout(rect=(0, 0.06, 1, 0.86))
    fig.savefig(OUT_PATH, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print(f"Wrote {OUT_PATH}")
    for m in models:
        r = reasoning[m]
        if m in no_reasoning:
            nr = no_reasoning[m]
            print(
                f"  {MODEL_LABELS[m]}: no-reasoning {nr['accuracy_pct']:.1f}% (d'={nr['d_prime']:.2f}) -> "
                f"medium-reasoning {r['accuracy_pct']:.1f}% (d'={r['d_prime']:.2f}), "
                f"delta={r['accuracy_pct']-nr['accuracy_pct']:+.1f}pp, reasoning_cost=${r['total_cost_usd']:.2f}"
            )
        else:
            print(
                f"  {MODEL_LABELS[m]}: no no-reasoning pass (mandatory reasoning) -> "
                f"medium-reasoning {r['accuracy_pct']:.1f}% (d'={r['d_prime']:.2f}), "
                f"reasoning_cost=${r['total_cost_usd']:.2f}"
            )
    print(f"  heuristic: {heuristic['accuracy_pct']:.1f}%")
    print(f"\nTotal reasoning-pass cost: ${reasoning['_totals']['total_cost_usd']:.2f}")
    print(f"Total no-reasoning-pass cost: ${no_reasoning['_totals']['total_cost_usd']:.2f}")


if __name__ == "__main__":
    main()
