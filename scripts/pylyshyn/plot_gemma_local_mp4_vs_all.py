"""
Figure for `claude/2026_09/2026_09_14/TODO.md`: compares the locally-hosted
(department cluster) gemma-4-31b-it fed native mp4 (this session's new run,
transformers' own torchcodec-backed video pipeline) against three existing
series on the same pylyshyn n_objects={3,4,5} sweep (fast redirect/fast
speed, native video only) -- its own OpenRouter-hosted results, the prior
local run that instead fed pre-decoded video.npy frames + a hand-built
video_metadata dict ([[project-status-2026-09-10]]), and the
nearest-neighbor position heuristic. Sibling of
plot_gemma_local_vs_openrouter.py, extended to four series to answer
whether switching to native mp4 decoding closes the local-vs-OpenRouter
accuracy gap.

Data sources:
  OpenRouter (n_objects=3 lives in speed_sweep_models/ since that run
  covered both redirect x speed conditions; 4/5 have their own dedicated
  run-names -- same SOURCES convention as plot_nobjects_sweep.py):
    n_objects=3: results/pylyshyn/speed_sweep_models/results.csv
    n_objects=4: results/pylyshyn/nobjects_sweep_n4/results.csv
    n_objects=5: results/pylyshyn/nobjects_sweep_n5/results.csv
  Local, mp4 (new, all three n_objects):
    results/pylyshyn/gemma_local_mp4_nobjects_sweep/results.csv
  Local, npy (existing, all three n_objects):
    results/pylyshyn/gemma_local_nobjects_sweep/results.csv

The heuristic line is computed once, directly on the local mp4 run's own
saved ground_truth.json files (data/pylyshyn/gemma_local_nobjects_sweep/
trials/) -- motion/ground truth is fully deterministic from (n_objects,
seed, redirect_condition, speed_condition), so this is numerically
identical across all three model series (mp4, npy, and OpenRouter all share
the same trial seeds), no need to touch more than one trial dir.

Usage:
    uv run python scripts/pylyshyn/plot_gemma_local_mp4_vs_all.py
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
LOCAL_MP4_COLOR = "#eb6834"
LOCAL_NPY_COLOR = "#8a5fc2"
HEURISTIC_COLOR = "#d1332e"

OPENROUTER_MODEL = "google/gemma-4-31b-it"
REDIRECT_CONDITION = "fast"
SPEED_CONDITION = "fast"
VARIANT = "native"
N_OBJECTS_VALUES = [3, 4, 5]

OPENROUTER_SOURCES = {
    3: "results/pylyshyn/speed_sweep_models/results.csv",
    4: "results/pylyshyn/nobjects_sweep_n4/results.csv",
    5: "results/pylyshyn/nobjects_sweep_n5/results.csv",
}
LOCAL_MP4_RESULTS_CSV = "results/pylyshyn/gemma_local_mp4_nobjects_sweep/results.csv"
LOCAL_NPY_RESULTS_CSV = "results/pylyshyn/gemma_local_nobjects_sweep/results.csv"
LOCAL_TRIALS_DIR = "data/pylyshyn/gemma_local_nobjects_sweep/trials"
OUT_PATH = "results/pylyshyn/gemma_local_mp4_nobjects_sweep/accuracy_by_nobjects_mp4_vs_all.png"


def _parse_bool(s: str) -> bool | None:
    if s == "True":
        return True
    if s == "False":
        return False
    return None


def load_openrouter_rows(n_objects: int) -> list[dict]:
    with open(OPENROUTER_SOURCES[n_objects]) as f:
        return [
            r for r in csv.DictReader(f)
            if r["model"] == OPENROUTER_MODEL and int(r["n_objects"]) == n_objects
            and r["redirect_condition"] == REDIRECT_CONDITION and r["speed_condition"] == SPEED_CONDITION
            and r["variant"] == VARIANT
        ]


def load_local_rows(results_csv: str, n_objects: int) -> list[dict]:
    with open(results_csv) as f:
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
    local_mp4_ys, local_mp4_errs = [], []
    local_npy_ys, local_npy_errs = [], []
    heuristic_ys = []
    n_openrouter = n_local_mp4 = n_local_npy = 0

    for n_objects in N_OBJECTS_VALUES:
        or_rows = load_openrouter_rows(n_objects)
        acc, sem, n = accuracy_stats(or_rows)
        openrouter_ys.append(acc)
        openrouter_errs.append(sem)
        n_openrouter = max(n_openrouter, n)

        mp4_rows = load_local_rows(LOCAL_MP4_RESULTS_CSV, n_objects)
        acc, sem, n = accuracy_stats(mp4_rows)
        local_mp4_ys.append(acc)
        local_mp4_errs.append(sem)
        n_local_mp4 = max(n_local_mp4, n)

        npy_rows = load_local_rows(LOCAL_NPY_RESULTS_CSV, n_objects)
        acc, sem, n = accuracy_stats(npy_rows)
        local_npy_ys.append(acc)
        local_npy_errs.append(sem)
        n_local_npy = max(n_local_npy, n)

        h_stats = heuristic_accuracy_by_group(mp4_rows, LOCAL_TRIALS_DIR, lambda row: n_objects)
        h_acc, _h_sem, _h_n = h_stats.get(n_objects, (float("nan"), 0, 0))
        heuristic_ys.append(h_acc)

    ax.errorbar(
        N_OBJECTS_VALUES, openrouter_ys, yerr=openrouter_errs, marker="o", markersize=7, linewidth=2,
        color=OPENROUTER_COLOR, ecolor=OPENROUTER_COLOR, elinewidth=1.5, capsize=4,
        label=f"OpenRouter (n={n_openrouter}/point)", zorder=3,
    )
    ax.errorbar(
        N_OBJECTS_VALUES, local_mp4_ys, yerr=local_mp4_errs, marker="o", markersize=7, linewidth=2,
        color=LOCAL_MP4_COLOR, ecolor=LOCAL_MP4_COLOR, elinewidth=1.5, capsize=4,
        label=f"Local cluster, mp4 (n={n_local_mp4}/point)", zorder=3,
    )
    ax.errorbar(
        N_OBJECTS_VALUES, local_npy_ys, yerr=local_npy_errs, marker="o", markersize=7, linewidth=2,
        color=LOCAL_NPY_COLOR, ecolor=LOCAL_NPY_COLOR, elinewidth=1.5, capsize=4,
        label=f"Local cluster, npy (n={n_local_npy}/point)", zorder=3,
    )
    ax.plot(
        N_OBJECTS_VALUES, heuristic_ys, marker="s", markersize=6, linewidth=1.5,
        linestyle="--", color=HEURISTIC_COLOR, zorder=2, label="Nearest-neighbor heuristic",
    )

    ax.set_xticks(N_OBJECTS_VALUES)
    ax.set_xlabel("n_objects", color=INK_SECONDARY)
    ax.set_ylabel("Accuracy (%)", color=INK_SECONDARY)
    fig.suptitle(
        "pylyshyn: gemma-4-31b-it, local mp4 vs. local npy vs. OpenRouter\n"
        "(fast redirect/fast speed, native only; error bars: SEM; gray dashed: chance)",
        color=INK_PRIMARY, fontsize=11, y=1.0,
    )
    ax.legend(frameon=False, labelcolor=INK_PRIMARY, fontsize=9.5, loc="lower left")

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT_PATH, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
