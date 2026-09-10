"""
Figure for `claude/2026_09/2026_09_10/TODO.md`: compares the locally-hosted
(department cluster) gemma-4-31b-it against its own OpenRouter-hosted
results on the pylyshyn n_objects={3,4,5} sweep (fast redirect/fast speed,
native video only), with the nearest-neighbor position heuristic
(`nearest_neighbor_heuristic.py`, see [[project-status-2026-09-01]])
overlaid as a third series -- analogous to
`results/pylyshyn/nobjects_sweep/accuracy_by_nobjects_fast.png`'s per-model
panels, but as one panel with local vs. OpenRouter vs. heuristic instead of
one panel per model.

OpenRouter data sources (n_objects=3 lives in speed_sweep_models/ since that
run covered both redirect x speed conditions; 4/5 have their own dedicated
run-names -- same SOURCES convention as plot_nobjects_sweep.py):
  n_objects=3: results/pylyshyn/speed_sweep_models/results.csv
  n_objects=4: results/pylyshyn/nobjects_sweep_n4/results.csv
  n_objects=5: results/pylyshyn/nobjects_sweep_n5/results.csv
Local data source (all three n_objects): results/pylyshyn/
  gemma_local_nobjects_sweep/results.csv (written by
  aggregate_gemma_local_nobjects_sweep.py).

The heuristic line is computed once, directly on the local run's own saved
ground_truth.json files (data/pylyshyn/gemma_local_nobjects_sweep/trials/) --
motion/ground truth is fully deterministic from (n_objects, seed,
redirect_condition, speed_condition), so this is numerically identical to
computing it on the OpenRouter trials at the same (n_objects, seed), no need
to touch both trial dirs.

Usage:
    uv run python scripts/pylyshyn/plot_gemma_local_vs_openrouter.py
"""

import csv
import math
import os

import matplotlib.pyplot as plt

from nearest_neighbor_heuristic import heuristic_accuracy_by_group

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
OPENROUTER_COLOR = "#2a78d6"
LOCAL_COLOR = "#eb6834"
HEURISTIC_COLOR = "#d1332e"

OPENROUTER_MODEL = "google/gemma-4-31b-it"
REDIRECT_CONDITION = "fast"
SPEED_CONDITION = "fast"
VARIANT = "native"
N_OBJECTS_VALUES = [3, 4, 5]

OPENROUTER_SOURCES = {
    3: ("results/pylyshyn/speed_sweep_models/results.csv", "data/pylyshyn/speed_sweep_models/trials"),
    4: ("results/pylyshyn/nobjects_sweep_n4/results.csv", "data/pylyshyn/nobjects_sweep_n4/trials"),
    5: ("results/pylyshyn/nobjects_sweep_n5/results.csv", "data/pylyshyn/nobjects_sweep_n5/trials"),
}
LOCAL_RESULTS_CSV = "results/pylyshyn/gemma_local_nobjects_sweep/results.csv"
LOCAL_TRIALS_DIR = "data/pylyshyn/gemma_local_nobjects_sweep/trials"
OUT_PATH = "results/pylyshyn/gemma_local_nobjects_sweep/accuracy_by_nobjects.png"


def _parse_bool(s: str) -> bool | None:
    if s == "True":
        return True
    if s == "False":
        return False
    return None


def load_openrouter_rows(n_objects: int) -> list[dict]:
    results_csv, _ = OPENROUTER_SOURCES[n_objects]
    with open(results_csv) as f:
        return [
            r for r in csv.DictReader(f)
            if r["model"] == OPENROUTER_MODEL and int(r["n_objects"]) == n_objects
            and r["redirect_condition"] == REDIRECT_CONDITION and r["speed_condition"] == SPEED_CONDITION
            and r["variant"] == VARIANT
        ]


def load_local_rows(n_objects: int) -> list[dict]:
    with open(LOCAL_RESULTS_CSV) as f:
        return [r for r in csv.DictReader(f) if int(r["n_objects"]) == n_objects]


def accuracy_stats(rows: list[dict]) -> tuple[float, float, int]:
    n = len(rows)
    correct = sum(1 for r in rows if _parse_bool(r["predicted"]) == _parse_bool(r["probe_is_target"]))
    p = correct / n
    sem = math.sqrt(p * (1 - p) / n) * 100
    return p * 100, sem, n


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.5, 6), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, color=GRIDLINE, linewidth=1, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED)
    ax.set_ylim(-5, 108)
    ax.axhline(50, color=INK_MUTED, linewidth=1, linestyle="--", zorder=1, label="_chance")

    openrouter_ys, openrouter_errs = [], []
    local_ys, local_errs = [], []
    heuristic_ys = []
    n_openrouter = n_local = 0

    for n_objects in N_OBJECTS_VALUES:
        or_rows = load_openrouter_rows(n_objects)
        acc, sem, n = accuracy_stats(or_rows)
        openrouter_ys.append(acc)
        openrouter_errs.append(sem)
        n_openrouter = max(n_openrouter, n)

        local_rows = load_local_rows(n_objects)
        acc, sem, n = accuracy_stats(local_rows)
        local_ys.append(acc)
        local_errs.append(sem)
        n_local = max(n_local, n)

        h_stats = heuristic_accuracy_by_group(local_rows, LOCAL_TRIALS_DIR, lambda row: n_objects)
        h_acc, _h_sem, _h_n = h_stats.get(n_objects, (float("nan"), 0, 0))
        heuristic_ys.append(h_acc)

    ax.errorbar(
        N_OBJECTS_VALUES, openrouter_ys, yerr=openrouter_errs, marker="o", markersize=7, linewidth=2,
        color=OPENROUTER_COLOR, ecolor=OPENROUTER_COLOR, elinewidth=1.5, capsize=4,
        label=f"OpenRouter (n={n_openrouter}/point)", zorder=3,
    )
    ax.errorbar(
        N_OBJECTS_VALUES, local_ys, yerr=local_errs, marker="o", markersize=7, linewidth=2,
        color=LOCAL_COLOR, ecolor=LOCAL_COLOR, elinewidth=1.5, capsize=4,
        label=f"Local cluster (n={n_local}/point)", zorder=3,
    )
    ax.plot(
        N_OBJECTS_VALUES, heuristic_ys, marker="s", markersize=6, linewidth=1.5,
        linestyle="--", color=HEURISTIC_COLOR, zorder=2, label="Nearest-neighbor heuristic",
    )

    ax.set_xticks(N_OBJECTS_VALUES)
    ax.set_xlabel("n_objects", color=INK_SECONDARY)
    ax.set_ylabel("Accuracy (%)", color=INK_SECONDARY)
    fig.suptitle(
        "pylyshyn: gemma-4-31b-it, local cluster vs. OpenRouter\n"
        "(fast redirect/fast speed, native only; error bars: SEM; gray dashed: chance)",
        color=INK_PRIMARY, fontsize=11, y=1.0,
    )
    ax.legend(frameon=False, labelcolor=INK_PRIMARY, fontsize=9.5, loc="lower left")

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT_PATH, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
