"""
Bar chart for `claude/2026_09/2026_09_14/TODO.md`'s Task 2: compares local
gemma-4-31b-it accuracy (npy vs. mp4 input pipelines) against the
nearest-neighbor position heuristic, on the 100-trial n_objects=4 sample
deliberately constructed so the heuristic sits at chance
(`generate_gemma_local_n4_heuristic_chance_stimuli.py`). Follow-up to
today's earlier 60-trial finding that mp4 input dramatically improved
accuracy at n=4 -- this checks whether that holds at higher power once the
heuristic's above-chance edge on the original sample is neutralized.

Error bars are SEM (`sqrt(p*(1-p)/n)*100`), same convention as every other
accuracy plot in scripts/pylyshyn/ -- an earlier version of this script used
the wider population Bernoulli std (`sqrt(p*(1-p))*100`) instead, per an
initial request, then was switched back to SEM per explicit follow-up
request. Bar-plus-errorbar structure mirrors `plot_stretch_sweep.py`'s
`ax.bar(...)` + `ax.errorbar(..., fmt="none")` overlay; styling constants
shared with every other `plot_*.py` in this directory.

Data sources:
  Local, npy:  results/pylyshyn/gemma_local_n4_heuristic_chance/npy/results.csv
  Local, mp4:  results/pylyshyn/gemma_local_n4_heuristic_chance/mp4/results.csv
  Heuristic:   computed directly via nearest_neighbor_heuristic
               .heuristic_accuracy_by_group on the mp4 arm's rows (ground
               truth is identical between the npy/mp4 arms -- same trial
               dirs), std then derived from that accuracy the same way as
               the two model bars.

Usage:
    uv run python scripts/pylyshyn/plot_gemma_local_n4_heuristic_chance_bar.py
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

BAR_COLORS = {"npy": "#8a5fc2", "mp4": "#eb6834", "heuristic": "#d1332e"}
BAR_LABELS = {"npy": "Local, npy", "mp4": "Local, mp4", "heuristic": "Nearest-neighbor heuristic"}

NPY_RESULTS_CSV = "results/pylyshyn/gemma_local_n4_heuristic_chance/npy/results.csv"
MP4_RESULTS_CSV = "results/pylyshyn/gemma_local_n4_heuristic_chance/mp4/results.csv"
TRIALS_DIR = "data/pylyshyn/gemma_local_n4_heuristic_chance/trials"
OUT_PATH = "results/pylyshyn/gemma_local_n4_heuristic_chance/accuracy_bar_npy_vs_mp4_vs_heuristic.png"


def _parse_bool(s: str) -> bool | None:
    if s == "True":
        return True
    if s == "False":
        return False
    return None


def accuracy_and_sem(rows: list[dict]) -> tuple[float, float, int]:
    n = len(rows)
    correct = sum(1 for r in rows if _parse_bool(r["predicted"]) == _parse_bool(r["probe_is_target"]))
    p = correct / n
    sem = math.sqrt(p * (1 - p) / n) * 100
    return p * 100, sem, n


def load_rows(results_csv: str) -> list[dict]:
    with open(results_csv) as f:
        return list(csv.DictReader(f))


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    npy_rows = load_rows(NPY_RESULTS_CSV)
    mp4_rows = load_rows(MP4_RESULTS_CSV)

    npy_acc, npy_sem, npy_n = accuracy_and_sem(npy_rows)
    mp4_acc, mp4_sem, mp4_n = accuracy_and_sem(mp4_rows)

    h_stats = heuristic_accuracy_by_group(mp4_rows, TRIALS_DIR, lambda row: "n4")
    h_acc, h_sem, h_n = h_stats["n4"]

    bars = [
        ("npy", npy_acc, npy_sem, npy_n),
        ("mp4", mp4_acc, mp4_sem, mp4_n),
        ("heuristic", h_acc, h_sem, h_n),
    ]

    fig, ax = plt.subplots(figsize=(6.5, 6), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, axis="y", color=GRIDLINE, linewidth=1, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED)
    ax.set_ylim(0, 118)
    ax.axhline(50, color=INK_MUTED, linewidth=1, linestyle="--", zorder=1)

    xs = range(len(bars))
    heights = [b[1] for b in bars]
    sems = [b[2] for b in bars]
    colors = [BAR_COLORS[b[0]] for b in bars]
    ax.bar(xs, heights, width=0.55, color=colors, zorder=3, edgecolor="#fcfcfb", linewidth=2)
    ax.errorbar(
        xs, heights, yerr=sems, fmt="none", ecolor=INK_PRIMARY, elinewidth=1.5, capsize=5, zorder=4,
    )

    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"{BAR_LABELS[b[0]]}\n(n={b[3]})" for b in bars])
    ax.set_xlim(-0.6, len(bars) - 0.4)
    ax.set_ylabel("Accuracy (%)", color=INK_SECONDARY)

    fig.suptitle(
        "pylyshyn n_objects=4: local gemma-4-31b-it, npy vs. mp4 input,\n"
        "heuristic neutralized to chance by construction\n"
        "(error bars: SEM; dashed line: chance)",
        color=INK_PRIMARY, fontsize=11, y=1.0,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(OUT_PATH, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print(f"Wrote {OUT_PATH}")
    for name, acc, sem, n in bars:
        print(f"  {BAR_LABELS[name]}: {acc:.1f}% (sem={sem:.1f}%, n={n})")


if __name__ == "__main__":
    main()
