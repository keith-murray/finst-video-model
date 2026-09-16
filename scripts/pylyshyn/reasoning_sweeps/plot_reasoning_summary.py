"""
Summary figure for claude/2026_08_20/TODO.md's reasoning-budget sweep
(results/pylyshyn/reasoning_sweeps/reasoning_sweep/results.csv, produced by
run_pylyshyn_reasoning_batch.py): percent error (100 - accuracy, an
unparseable answer counts as wrong) on the y-axis, reasoning effort level on
the x-axis, one line per model -- per the TODO's exact spec ("x-axis is
reasoning budget ... y axis is percent error").

x-axis is ordinal (REASONING_LEVEL_ORDER from the batch runner: none, minimal,
low, medium, high), not a numeric token count -- OpenRouter's `reasoning.
max_tokens` turned out not to be an enforced budget (a pilot found 654
reasoning tokens spent against a requested cap of 100), so `effort` is the
actual requestable "budget" tier and the only fair shared axis across models.
qwen3.8-27b has a "none" point (reasoning disabled) that qwen3.8-max lacks
(reasoning is mandatory there), so its line is one point shorter/shifted
right on the shared axis.

Colors/chrome follow scripts/pylyshyn/reasoning_sweeps/plot_qwen_summary.py's convention
(validated categorical slots 1-2: blue, orange).

Usage:
    uv run python scripts/pylyshyn/reasoning_sweeps/plot_reasoning_summary.py
"""

import csv
from collections import defaultdict

import matplotlib.pyplot as plt

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES_COLORS = ["#2a78d6", "#eb6834"]  # blue, orange -- validated categorical slots 1-2

RESULTS_CSV = "results/pylyshyn/reasoning_sweeps/reasoning_sweep/results.csv"
REASONING_LEVEL_ORDER = ["none", "minimal", "low", "medium", "high"]
MODEL_LABELS = {
    "qwen/qwen3.8-27b": "qwen3.8-27b",
    "qwen/qwen3.8-max": "qwen3.8-max",
}
OUT_PATH = "results/pylyshyn/reasoning_sweeps/reasoning_sweep/reasoning_sweep_percent_error.png"


def _parse_bool(s: str) -> bool | None:
    if s == "True":
        return True
    if s == "False":
        return False
    return None


def percent_error_by_model_level(results_csv: str) -> dict[tuple[str, str], float]:
    totals = defaultdict(int)
    wrong = defaultdict(int)
    with open(results_csv) as f:
        for row in csv.DictReader(f):
            key = (row["model"], row["reasoning_level"])
            totals[key] += 1
            probe_is_target = _parse_bool(row["probe_is_target"])
            predicted = _parse_bool(row["predicted"])
            if predicted != probe_is_target:
                wrong[key] += 1
    return {key: 100 * wrong[key] / total for key, total in totals.items()}


def main():
    error_by_model_level = percent_error_by_model_level(RESULTS_CSV)
    models = sorted({model for model, _ in error_by_model_level}, key=list(MODEL_LABELS).index)

    fig, ax = plt.subplots(figsize=(7.5, 5.5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, color=GRIDLINE, linewidth=1, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED)
    ax.set_ylim(-5, 65)
    ax.axhline(50, color=INK_MUTED, linewidth=1, linestyle="--", zorder=1, label="_chance")

    for model, color in zip(models, SERIES_COLORS):
        points = sorted(
            (REASONING_LEVEL_ORDER.index(level), level, error)
            for (m, level), error in error_by_model_level.items() if m == model
        )
        xs = [x for x, _, _ in points]
        ys = [e for _, _, e in points]
        ax.plot(
            xs, ys, marker="o", markersize=8, linewidth=2,
            color=color, label=MODEL_LABELS[model], zorder=3,
        )

    ax.set_xticks(range(len(REASONING_LEVEL_ORDER)))
    ax.set_xticklabels(REASONING_LEVEL_ORDER)
    ax.set_xlim(-0.3, len(REASONING_LEVEL_ORDER) - 0.7)
    ax.set_xlabel("Reasoning budget (effort level)", color=INK_SECONDARY)
    ax.set_ylabel("Percent error (%)", color=INK_SECONDARY)
    ax.set_title(
        "Reasoning budget vs. accuracy on the simplest pylyshyn condition\n"
        "(n_objects=2, n_cued=1)",
        color=INK_PRIMARY, fontsize=12, fontweight="bold",
    )

    ax.legend(frameon=False, labelcolor=INK_PRIMARY, fontsize=10, loc="upper right")

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150, facecolor=fig.get_facecolor())
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
