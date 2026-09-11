"""
Figure for `claude/2026_09/2026_09_10/TODO.md`'s "Testing different hosts of
gemma4" task: does google/gemma-4-31b-it's debug_circular accuracy
(rotation_deg 40/80/120, native only) vary across the 7 OpenRouter hosts
pinned by `run_debug_circular_host_sweep.py`? One line per host, per the
TODO's "on the figures, just note the host" -- quantization/other host info
goes in the written summary instead, not the legend.

Usage:
    uv run python scripts/debug_circular/plot_host_sweep.py
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

# 7-color categorical palette (validated for light/dark legibility).
HOST_COLORS = {
    "deepinfra-fp4": "#2a78d6",
    "deepinfra-fp8": "#6aa8e8",
    "coreweave": "#eb6834",
    "crusoe": "#2ba84a",
    "parasail": "#d1332e",
    "together": "#9b59d0",
    "modelrun": "#a0824a",
}

RESULTS_CSV = "results/debug_circular/host_sweep/results.csv"
OUT_PATH = "results/debug_circular/host_sweep/accuracy_by_angle_hosts.png"


def _parse_bool(s: str) -> bool | None:
    if s == "True":
        return True
    if s == "False":
        return False
    return None


def accuracy_by_group(results_csv: str) -> dict[tuple[float, str], tuple[float, float, int]]:
    """Returns {(rotation_deg, host): (accuracy_pct, sem_pct, n)}."""
    totals = defaultdict(int)
    correct = defaultdict(int)
    with open(results_csv) as f:
        for row in csv.DictReader(f):
            if row["predicted"] == "":
                continue
            key = (float(row["rotation_deg"]), row["host"])
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
    rotation_degs = sorted({k[0] for k in stats})
    hosts_present = [h for h in HOST_COLORS if any(k[1] == h for k in stats)]

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
        for rd in rotation_degs:
            acc, sem, n = stats.get((rd, host), (float("nan"), 0, 0))
            ys.append(acc)
            errs.append(sem)
            n_per_point = max(n_per_point, n)
        ax.errorbar(
            rotation_degs, ys, yerr=errs, marker="o", markersize=6, linewidth=1.8,
            color=HOST_COLORS[host], ecolor=HOST_COLORS[host],
            elinewidth=1.2, capsize=3, label=host, zorder=3,
        )

    ax.set_xticks(rotation_degs)
    ax.set_xlabel("rotation_deg", color=INK_SECONDARY)
    ax.set_ylabel("Accuracy (%)", color=INK_SECONDARY)

    fig.suptitle(
        "google/gemma-4-31b-it on debug_circular, by OpenRouter host\n"
        f"(n_objects=3, native only; error bars: SEM, n={n_per_point}/point; dashed line: chance)",
        color=INK_PRIMARY, fontsize=12, y=1.0,
    )
    ax.legend(frameon=False, labelcolor=INK_PRIMARY, fontsize=9.5, loc="lower left", ncol=2)

    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(OUT_PATH, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
