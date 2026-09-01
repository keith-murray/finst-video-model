"""
Figure for `claude/2026_09/2026_09_01/TODO.md`'s Task 3: does the
nearest-neighbor position heuristic's near-model-level pylyshyn accuracy
([[project-status-2026-09-01]], `scripts/pylyshyn/nearest_neighbor_heuristic
.py`) degrade as `n_objects` scales up, while genuine identity tracking
would not? First tried at slow-redirect/slow-speed (both models' strongest
condition) -- a null result, the heuristic didn't degrade at all
(replotted here on request as `--condition slow`).

`--condition fast` (added same day, same TODO) retries the same sweep at
fast-redirect/fast-speed instead, per the working explanation for the null
result: at slow motion, absolute displacement by probe time is small
regardless of `n_objects`, so how often a distractor randomly drifts near
the target's remembered original spot is governed by motion speed/
duration, not object count -- predicting the heuristic-vs-tracking gap (if
any) should be more visible under faster motion.

Data sources (three different run-names/data dirs per condition, since
each locks its own seed set -- see [[project-status-2026-09-01]] and
`run_pylyshyn_speed_sweep_models.py`'s module docstring for why these stay
separate rather than one shared results.csv):
  n_objects=3: results/pylyshyn/speed_sweep/results.csv (qwen3.6-plus, 16
    seeds) + results/pylyshyn/speed_sweep_models/results.csv (gemma-4-31b-it,
    20 seeds) -- both already contain all 4 redirect x speed conditions.
  n_objects=4: results/pylyshyn/nobjects_sweep_n4/results.csv (both models;
    the slow and fast conditions were run as two separate invocations under
    this same run-name -- safe since the resume key includes
    redirect_condition/speed_condition, though note config.json's own
    `redirect_conditions`/`speed_conditions` metadata only reflects
    whichever invocation ran first).
  n_objects=5: results/pylyshyn/nobjects_sweep_n5/results.csv (ditto).

Line plot per model (2 panels): x-axis n_objects in {3,4,5}, y-axis
accuracy (%), one line per variant (native/stretched) with SEM error bars,
plus a red dashed line per n_objects point for the heuristic's accuracy on
that exact trial set (computed via `nearest_neighbor_heuristic
.heuristic_accuracy_by_group`, no video/API calls).

Usage:
    uv run python scripts/pylyshyn/plot_nobjects_sweep.py
    uv run python scripts/pylyshyn/plot_nobjects_sweep.py --condition fast
"""

import argparse
import csv
import math
from collections import defaultdict

import matplotlib.pyplot as plt

from nearest_neighbor_heuristic import heuristic_accuracy_by_group

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
VARIANT_COLORS = {"native": "#2a78d6", "stretched": "#eb6834"}
VARIANT_LABELS = {"native": "Native (fps=10, ~10s)", "stretched": "Stretched (fps=2, ~50s)"}
HEURISTIC_COLOR = "#d1332e"

# (n_objects, model) -> (results_csv, data_trials_dir)
SOURCES = {
    (3, "qwen/qwen3.6-plus"): ("results/pylyshyn/speed_sweep/results.csv", "data/pylyshyn/speed_sweep/trials"),
    (3, "google/gemma-4-31b-it"): ("results/pylyshyn/speed_sweep_models/results.csv", "data/pylyshyn/speed_sweep_models/trials"),
    (4, "qwen/qwen3.6-plus"): ("results/pylyshyn/nobjects_sweep_n4/results.csv", "data/pylyshyn/nobjects_sweep_n4/trials"),
    (4, "google/gemma-4-31b-it"): ("results/pylyshyn/nobjects_sweep_n4/results.csv", "data/pylyshyn/nobjects_sweep_n4/trials"),
    (5, "qwen/qwen3.6-plus"): ("results/pylyshyn/nobjects_sweep_n5/results.csv", "data/pylyshyn/nobjects_sweep_n5/trials"),
    (5, "google/gemma-4-31b-it"): ("results/pylyshyn/nobjects_sweep_n5/results.csv", "data/pylyshyn/nobjects_sweep_n5/trials"),
}
N_OBJECTS_VALUES = [3, 4, 5]
MODELS = [("qwen/qwen3.6-plus", "qwen3.6-plus"), ("google/gemma-4-31b-it", "gemma-4-31b-it")]
CONDITIONS = {
    "slow": ("slow", "slow", "redirect_s=2.0, speed_px_s=16-33"),
    "fast": ("fast", "fast", "redirect_s=1.0, speed_px_s=32-66"),
}


def _parse_bool(s: str) -> bool | None:
    if s == "True":
        return True
    if s == "False":
        return False
    return None


def load_rows(results_csv: str, model: str, redirect_condition: str, speed_condition: str) -> list[dict]:
    with open(results_csv) as f:
        return [
            r for r in csv.DictReader(f)
            if r["model"] == model
            and r["redirect_condition"] == redirect_condition and r["speed_condition"] == speed_condition
        ]


def accuracy_stats(rows: list[dict], variant: str) -> tuple[float, float, int]:
    matched = [r for r in rows if r["variant"] == variant]
    n = len(matched)
    correct = sum(1 for r in matched if _parse_bool(r["predicted"]) == _parse_bool(r["probe_is_target"]))
    p = correct / n
    sem = math.sqrt(p * (1 - p) / n) * 100
    return p * 100, sem, n


def main():
    import os

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", type=str, default="slow", choices=list(CONDITIONS.keys()))
    args = parser.parse_args()
    redirect_condition, speed_condition, condition_label = CONDITIONS[args.condition]

    os.makedirs("results/pylyshyn/nobjects_sweep", exist_ok=True)
    out_path = f"results/pylyshyn/nobjects_sweep/accuracy_by_nobjects_{args.condition}.png"

    fig, axes = plt.subplots(1, len(MODELS), figsize=(8.5, 6), facecolor="#fcfcfb", sharey=True)
    fig.suptitle(
        f"pylyshyn: accuracy vs. n_objects ({condition_label})\n"
        "(error bars: SEM; gray dashed: chance; red dashed: nearest-neighbor heuristic)",
        color=INK_PRIMARY, fontsize=12, y=1.0,
    )

    for ax, (model, model_label) in zip(axes, MODELS):
        ax.set_facecolor("#fcfcfb")
        ax.grid(True, color=GRIDLINE, linewidth=1, zorder=0)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(BASELINE)
        ax.tick_params(colors=INK_MUTED)
        ax.set_ylim(-5, 108)
        ax.axhline(50, color=INK_MUTED, linewidth=1, linestyle="--", zorder=1, label="_chance")

        heuristic_ys = []
        n_per_point = 0
        for variant in ("native", "stretched"):
            ys, errs = [], []
            for n_objects in N_OBJECTS_VALUES:
                results_csv, _ = SOURCES[(n_objects, model)]
                rows = load_rows(results_csv, model, redirect_condition, speed_condition)
                acc, sem, n = accuracy_stats(rows, variant)
                ys.append(acc)
                errs.append(sem)
                n_per_point = max(n_per_point, n)
            ax.errorbar(
                N_OBJECTS_VALUES, ys, yerr=errs, marker="o", markersize=7, linewidth=2,
                color=VARIANT_COLORS[variant], ecolor=VARIANT_COLORS[variant],
                elinewidth=1.5, capsize=4, label=VARIANT_LABELS[variant], zorder=3,
            )

        for n_objects in N_OBJECTS_VALUES:
            results_csv, trials_dir = SOURCES[(n_objects, model)]
            rows = load_rows(results_csv, model, redirect_condition, speed_condition)
            h_stats = heuristic_accuracy_by_group(rows, trials_dir, lambda row: args.condition)
            h_acc, h_sem, h_n = h_stats.get(args.condition, (float("nan"), 0, 0))
            heuristic_ys.append(h_acc)
        ax.plot(
            N_OBJECTS_VALUES, heuristic_ys, marker="s", markersize=6, linewidth=1.5,
            linestyle="--", color=HEURISTIC_COLOR, zorder=2,
            label="Nearest-neighbor heuristic" if ax is axes[0] else "_heuristic",
        )

        ax.set_xticks(N_OBJECTS_VALUES)
        ax.set_xlabel("n_objects", color=INK_SECONDARY)
        ax.set_title(f"{model_label} (n={n_per_point}/point)", color=INK_PRIMARY, fontsize=11)

    axes[0].set_ylabel("Accuracy (%)", color=INK_SECONDARY)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, frameon=False, labelcolor=INK_PRIMARY, fontsize=10,
        loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02),
    )

    fig.tight_layout(rect=(0, 0.06, 1, 0.88))
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
