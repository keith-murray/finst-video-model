"""
Summary figure for the gemma-4-31b-it frame-budget sweep
(results/debug_circular/frame_budget_sweep/results.csv, produced by
run_gemma_frame_budget_sweep.py -- see claude/2026_09/2026_09_08/TODO.md):
mean `prompt_tokens` (line, with the individual rep points shown too) on the
y-axis vs. real frame count (1-15) on the x-axis. If gemma has a fixed
visual-token budget, this should rise roughly linearly while frame count is
below the budget, then flatten once frame count exceeds it -- the flattening
point is the thing this sweep exists to read off directly.

Colors/styling follow scripts/debug_circular/plot_openrouter_diagnostic.py's
palette convention (validated categorical slot 1: blue for the line/points).

Usage:
    uv run python scripts/debug_circular/plot_frame_budget_sweep.py
"""

import csv
from collections import defaultdict

import matplotlib.pyplot as plt

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
LINE_COLOR = "#2a78d6"  # validated categorical slot 1

RESULTS_CSV = "results/debug_circular/frame_budget_sweep/results.csv"
OUT_PATH = "results/debug_circular/frame_budget_sweep/prompt_tokens_by_frame_count.png"


def tokens_by_frame_count(results_csv: str) -> dict[int, list[int]]:
    by_n = defaultdict(list)
    with open(results_csv) as f:
        for row in csv.DictReader(f):
            if row["prompt_tokens"]:
                by_n[int(row["n_frames"])].append(int(row["prompt_tokens"]))
    return by_n


def main():
    by_n = tokens_by_frame_count(RESULTS_CSV)
    n_frames_sorted = sorted(by_n)
    means = [sum(by_n[n]) / len(by_n[n]) for n in n_frames_sorted]

    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, axis="y", color=GRIDLINE, linewidth=1, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED)

    for n in n_frames_sorted:
        ax.scatter([n] * len(by_n[n]), by_n[n], color=LINE_COLOR, alpha=0.35, zorder=3, s=30)
    ax.plot(n_frames_sorted, means, color=LINE_COLOR, linewidth=2, marker="o", zorder=4)

    ax.set_xticks(n_frames_sorted)
    ax.set_xlabel("Real frame count", color=INK_SECONDARY)
    ax.set_ylabel("prompt_tokens", color=INK_SECONDARY)
    ax.set_title(
        "google/gemma-4-31b-it: prompt_tokens vs. real frame count (1-15)",
        color=INK_PRIMARY, fontsize=13, fontweight="bold",
    )
    fig.suptitle(
        "(points: individual reps; line: mean per frame count)",
        color=INK_MUTED, fontsize=10, y=0.93,
    )

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
