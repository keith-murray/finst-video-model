"""
Summary figure for claude/2026_08_19/TODO.md's Qwen 3.8 pylyshyn grid: percent
error (100 - accuracy) on the y-axis, number of cued objects (n_cued) on the
x-axis, one line per model, one panel per total object count (n_objects) --
5 panels for n_objects in {2, 4, 6, 8, 10}, each showing whichever n_cued
values that condition actually has (1 for n_objects in {2, 4}; 1, 3 for
{6, 8}; 1, 3, 5 for 10).

MODEL_RUNS has three lines, not two -- qwen3.8-27b appears twice, once per
reasoning setting, since the reasoning-effort comparison is itself part of
the story: "low" (results/pylyshyn/reasoning_sweeps/qwen3.8-27b-low/) vs. fully disabled
(results/pylyshyn/reasoning_sweeps/qwen3.8-27b/), where an 18-trial spot check found
disabling reasoning entirely crippled the model (d' -1.71 vs. +0.77 at
"low" on the same trials) rather than just saving money. qwen3.8-max only
has one usable setting (reasoning is mandatory, "minimal" is its floor),
results/pylyshyn/reasoning_sweeps/qwen3.8-max/.

Colors and figure chrome follow scripts/prototype/plot_summary_figure.py's
shared-legend-across-panels convention (validated categorical slots 1-3:
blue, orange, aqua).

Usage:
    uv run python scripts/pylyshyn/reasoning_sweeps/plot_qwen_summary.py
"""

import csv
from collections import defaultdict

import matplotlib.pyplot as plt

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # blue, orange, aqua -- validated categorical slots 1-3

MODEL_RUNS = [
    ("qwen/qwen3.8-27b", "results/pylyshyn/reasoning_sweeps/qwen3.8-27b-low/results.csv", "qwen3.8-27b (low reasoning)"),
    ("qwen/qwen3.8-max", "results/pylyshyn/reasoning_sweeps/qwen3.8-max/results.csv", "qwen3.8-max (minimal reasoning)"),
    ("qwen/qwen3.8-27b", "results/pylyshyn/reasoning_sweeps/qwen3.8-27b/results.csv", "qwen3.8-27b (no reasoning)"),
]
N_OBJECTS_PANELS = [2, 4, 6, 8, 10]
OUT_PATH = "results/pylyshyn/reasoning_sweeps/qwen38_summary_percent_error.png"


def _parse_bool(s: str) -> bool | None:
    if s == "True":
        return True
    if s == "False":
        return False
    return None


def percent_error_by_condition(results_csv: str) -> dict[tuple[int, int], float]:
    """(n_objects, n_cued) -> percent of trials the model got wrong (an
    unparseable answer counts as wrong, since it's not a correct answer)."""
    totals = defaultdict(int)
    wrong = defaultdict(int)
    with open(results_csv) as f:
        for row in csv.DictReader(f):
            key = (int(row["n_objects"]), int(row["n_cued"]))
            totals[key] += 1
            probe_is_target = _parse_bool(row["probe_is_target"])
            predicted = _parse_bool(row["predicted"])
            if predicted != probe_is_target:
                wrong[key] += 1
    return {key: 100 * wrong[key] / total for key, total in totals.items()}


def main():
    per_model_error = {
        label: percent_error_by_condition(results_csv)
        for _, results_csv, label in MODEL_RUNS
    }
    color_by_label = dict(zip((label for _, _, label in MODEL_RUNS), SERIES_COLORS))

    fig, axes = plt.subplots(1, len(N_OBJECTS_PANELS), figsize=(18, 4.2), facecolor="#fcfcfb", sharey=True)
    fig.suptitle(
        "Qwen 3.8 on the pylyshyn task: percent error by cue count and field size",
        color=INK_PRIMARY, fontsize=13, fontweight="bold", y=1.08,
    )

    for ax, n_objects in zip(axes, N_OBJECTS_PANELS):
        ax.set_facecolor("#fcfcfb")
        ax.grid(True, color=GRIDLINE, linewidth=1, zorder=0)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(BASELINE)
        ax.tick_params(colors=INK_MUTED)
        ax.set_ylim(-5, 105)
        ax.axhline(50, color=INK_MUTED, linewidth=1, linestyle="--", zorder=1, label="_chance")

        all_n_cued = sorted({
            n_cued for error in per_model_error.values()
            for (n_obj, n_cued) in error if n_obj == n_objects
        })
        ax.set_xticks(all_n_cued)
        pad = (max(all_n_cued) - min(all_n_cued)) * 0.3 if len(all_n_cued) > 1 else 1.0
        ax.set_xlim(min(all_n_cued) - pad, max(all_n_cued) + pad)

        for _, _, label in MODEL_RUNS:
            error = per_model_error[label]
            points = sorted((n_cued, error[(n_objects, n_cued)]) for (n_obj, n_cued) in error if n_obj == n_objects)
            if not points:
                continue
            xs = [x for x, _ in points]
            ys = [y for _, y in points]
            ax.plot(
                xs, ys, marker="o", markersize=8, linewidth=2,
                color=color_by_label[label], label=label, zorder=3,
            )

        ax.set_xlabel("n_cued", color=INK_SECONDARY)
        ax.set_title(f"n_objects = {n_objects}", color=INK_PRIMARY, fontsize=11)

    axes[0].set_ylabel("Percent error (%)", color=INK_SECONDARY)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0),
        ncol=len(MODEL_RUNS), frameon=False, labelcolor=INK_PRIMARY, fontsize=10,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(OUT_PATH, dpi=150, facecolor=fig.get_facecolor())
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
