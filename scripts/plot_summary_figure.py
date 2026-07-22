"""
Two-panel PI summary figure combining the headline "exact identity match"
panel from two already-run sweeps:
  - left:  accuracy vs. n_circles, one line per speed_px_s (results/e8489fbd)
  - right: accuracy vs. tracking_s, one line per speed_px_s (results/ce753d46)

Both sweeps used the same five speed_px_s values, so the two panels share one
color legend across the top instead of each plotting its own.

Reuses load_rows/aggregate/SERIES_COLORS from plot_comprehension_results.py
rather than duplicating the aggregation or palette logic.

Usage:
    uv run python scripts/plot_summary_figure.py
"""

import matplotlib.pyplot as plt

from plot_comprehension_results import (
    load_rows, aggregate, SERIES_COLORS, INK_PRIMARY, INK_SECONDARY,
    INK_MUTED, GRIDLINE, BASELINE,
)

PANELS = [
    ("results/e8489fbd/results.csv", "n_circles", "Accuracy vs. object count"),
    ("results/ce753d46/results.csv", "tracking_s", "Accuracy vs. tracking duration (s)"),
]
OUT_PATH = "results/summary_capacity_and_duration.png"


def main():
    summaries = [
        (aggregate(load_rows(csv_path), x_axis, "speed_px_s"), x_axis, title)
        for csv_path, x_axis, title in PANELS
    ]

    speed_values = sorted(
        {s for summary, _, _ in summaries for s, _ in summary}, key=float
    )
    color_by_speed = dict(zip(speed_values, SERIES_COLORS))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), facecolor="#fcfcfb")

    for ax, (summary, x_axis, title) in zip(axes, summaries):
        ax.set_facecolor("#fcfcfb")
        ax.grid(True, color=GRIDLINE, linewidth=1, zorder=0)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(BASELINE)
        ax.tick_params(colors=INK_MUTED)
        ax.set_ylim(-0.05, 1.05)

        all_x = sorted({x for _, x in summary})
        ax.set_xticks(all_x)
        pad = (max(all_x) - min(all_x)) * 0.08 if len(all_x) > 1 else 1.0
        ax.set_xlim(min(all_x) - pad, max(all_x) + pad)

        for speed in speed_values:
            points = sorted((x, s) for (spd, x), s in summary.items() if spd == speed)
            if not points:
                continue
            xs = [x for x, _ in points]
            ax.errorbar(
                xs, [s["exact_match_rate"] for _, s in points],
                yerr=[s["exact_match_sem"] for _, s in points],
                marker="o", markersize=8, linewidth=2, capsize=3,
                color=color_by_speed[speed], label=f"speed_px_s={speed}", zorder=3,
            )

        ax.set_xlabel(x_axis, color=INK_SECONDARY)
        ax.set_ylabel("Fraction of trials correct", color=INK_SECONDARY)
        ax.set_title(title, color=INK_PRIMARY, fontsize=11)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02),
        ncol=len(speed_values), frameon=False, labelcolor=INK_PRIMARY, fontsize=9,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(OUT_PATH, dpi=150)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
