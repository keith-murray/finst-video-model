"""
Figure for `claude/2026_09/2026_09_10/TODO.md`'s "Testing different hosts of
gemma4" task: does google/gemma-4-31b-it's pylyshyn accuracy (n_objects
3/4/5, fast redirect/fast speed, native only) vary across the 7 OpenRouter
hosts pinned by `run_pylyshyn_host_sweep.py`? One line per host (per the
TODO's "on the figures, just note the host"), plus the nearest-neighbor
position heuristic (dashed) -- computed once via
`nearest_neighbor_heuristic.heuristic_accuracy_by_group` on the sweep's own
ground truth, since motion/ground truth is host-independent (deterministic
from seed), same "compute once" pattern as
`scripts/pylyshyn/plot_gemma_local_vs_openrouter.py`.

Usage:
    uv run python scripts/pylyshyn/plot_host_sweep.py
"""

import csv
import json
import math
import os
from collections import defaultdict

import matplotlib.pyplot as plt

from nearest_neighbor_heuristic import heuristic_predicts_match

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
HEURISTIC_COLOR = "#d1332e"

HOST_COLORS = {
    "deepinfra-fp4": "#2a78d6",
    "deepinfra-fp8": "#6aa8e8",
    "coreweave": "#eb6834",
    "crusoe": "#2ba84a",
    "parasail": "#c9455f",
    "together": "#9b59d0",
    "modelrun": "#a0824a",
}

N_OBJECTS_VALUES = [3, 4, 5]
RESULTS_CSV = "results/pylyshyn/host_sweep/results.csv"
TRIALS_DIR = "data/pylyshyn/host_sweep/trials"
OUT_PATH = "results/pylyshyn/host_sweep/accuracy_by_nobjects_hosts.png"


def _parse_bool(s: str) -> bool | None:
    if s == "True":
        return True
    if s == "False":
        return False
    return None


def load_rows(n_objects: int, host: str | None = None) -> list[dict]:
    with open(RESULTS_CSV) as f:
        return [
            r for r in csv.DictReader(f)
            if int(r["n_objects"]) == n_objects and r["predicted"] != ""
            and (host is None or r["host"] == host)
        ]


def accuracy_stats(rows: list[dict]) -> tuple[float, float, int]:
    n = len(rows)
    correct = sum(1 for r in rows if _parse_bool(r["predicted"]) == _parse_bool(r["probe_is_target"]))
    p = correct / n
    sem = math.sqrt(p * (1 - p) / n) * 100
    return p * 100, sem, n


def heuristic_stats(n_objects: int) -> tuple[float, float, int]:
    """Ground truth is identical across hosts for a given (n_objects, seed,
    probe_on_target) -- dedup across the 7 host repeats and compute once.
    Trial dirs here are named `<host>_<trial_id>` (run_pylyshyn_host_sweep.py),
    not the bare `<trial_id>` nearest_neighbor_heuristic.heuristic_accuracy_by_group
    assumes, so this reimplements its lookup rather than reusing it."""
    seen = set()
    total = correct = 0
    for row in load_rows(n_objects):
        key = (int(row["seed"]), row["probe_on_target"] == "True")
        if key in seen:
            continue
        seen.add(key)
        gt_path = os.path.join(TRIALS_DIR, f"{row['host']}_{row['trial_id']}", "ground_truth.json")
        with open(gt_path) as f:
            ground_truth = json.load(f)
        total += 1
        correct += heuristic_predicts_match(ground_truth) == ground_truth["probe_is_target"]
    p = correct / total
    sem = math.sqrt(p * (1 - p) / total) * 100
    return p * 100, sem, total


def main():
    with open(RESULTS_CSV) as f:
        hosts_present = sorted({r["host"] for r in csv.DictReader(f)}, key=lambda h: list(HOST_COLORS).index(h))

    fig, ax = plt.subplots(figsize=(9, 6.5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, color=GRIDLINE, linewidth=1, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED)
    ax.set_ylim(-5, 108)
    ax.axhline(50, color=INK_MUTED, linewidth=1, linestyle="--", zorder=1, label="_chance")

    n_per_point = 0
    for host in hosts_present:
        ys, errs = [], []
        for n_objects in N_OBJECTS_VALUES:
            acc, sem, n = accuracy_stats(load_rows(n_objects, host))
            ys.append(acc)
            errs.append(sem)
            n_per_point = max(n_per_point, n)
        ax.errorbar(
            N_OBJECTS_VALUES, ys, yerr=errs, marker="o", markersize=6, linewidth=1.8,
            color=HOST_COLORS[host], ecolor=HOST_COLORS[host],
            elinewidth=1.2, capsize=3, label=host, zorder=3,
        )

    heuristic_ys = []
    for n_objects in N_OBJECTS_VALUES:
        h_acc, _h_sem, _h_n = heuristic_stats(n_objects)
        heuristic_ys.append(h_acc)
    ax.plot(
        N_OBJECTS_VALUES, heuristic_ys, marker="s", markersize=6, linewidth=1.5,
        linestyle="--", color=HEURISTIC_COLOR, zorder=2, label="Nearest-neighbor heuristic",
    )

    ax.set_xticks(N_OBJECTS_VALUES)
    ax.set_xlabel("n_objects", color=INK_SECONDARY)
    ax.set_ylabel("Accuracy (%)", color=INK_SECONDARY)

    fig.suptitle(
        "google/gemma-4-31b-it on pylyshyn, by OpenRouter host\n"
        "(fast redirect/fast speed, native only; "
        f"error bars: SEM, n={n_per_point}/point; dashed gray: chance)",
        color=INK_PRIMARY, fontsize=12, y=1.0,
    )
    ax.legend(frameon=False, labelcolor=INK_PRIMARY, fontsize=9.5, loc="lower left", ncol=2)

    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(OUT_PATH, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
