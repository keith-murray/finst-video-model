"""
Reads a batch runner's results.csv (either arm -- both share the
probe_is_target/predicted/outcome columns produced by
finst_video_model.scoring) and renders a two-panel figure:
a 2x2 confusion matrix (target/distractor probe x model's True/False
answer) and a hit-rate/false-alarm-rate bar panel annotated with d' and
criterion (finst_video_model.scoring.compute_d_prime).

Colors follow this project's dataviz convention: a single-hue sequential
blue ramp for the confusion matrix (magnitude = trial count per cell), and
the first two fixed categorical slots (blue, orange) for the two rates in
the right panel -- two distinct quantities, not a magnitude scale, so they
get identity color rather than a ramp.

Usage:
    uv run python scripts/plot_confusion_and_dprime.py \\
        results/<batch_id>/results.csv --title "Pylyshyn (1/10 cued)"
"""

import argparse
import csv

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from finst_video_model.scoring import compute_d_prime

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
CATEGORICAL_1 = "#2a78d6"  # blue -- hit rate
CATEGORICAL_2 = "#eb6834"  # orange -- false-alarm rate
SEQUENTIAL_BLUE = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]


def _parse_bool(s: str) -> bool | None:
    if s == "True":
        return True
    if s == "False":
        return False
    return None


def load_trials(results_csv: str) -> list[dict]:
    trials = []
    with open(results_csv) as f:
        for row in csv.DictReader(f):
            trials.append({
                "probe_is_target": _parse_bool(row["probe_is_target"]),
                "predicted": _parse_bool(row["predicted"]),
            })
    return trials


def plot(results_csv: str, title: str, out_path: str):
    trials = load_trials(results_csv)
    d = compute_d_prime(trials)

    fig, (ax_cm, ax_rates) = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.patch.set_facecolor("#fcfcfb")
    fig.suptitle(title, color=INK_PRIMARY, fontsize=13, fontweight="bold")

    # --- Confusion matrix ---
    matrix = [
        [d["hits"], d["misses"]],
        [d["false_alarms"], d["correct_rejections"]],
    ]
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL_BLUE)
    vmax = max(max(row) for row in matrix) or 1
    ax_cm.imshow(matrix, cmap=cmap, vmin=0, vmax=vmax, aspect="equal")
    ax_cm.set_xticks([0, 1], ["Said True", "Said False"], color=INK_SECONDARY)
    ax_cm.set_yticks([0, 1], ["Probe = target", "Probe = distractor"], color=INK_SECONDARY)
    ax_cm.set_title("Confusion matrix", color=INK_PRIMARY, fontsize=11, pad=10)
    for spine in ax_cm.spines.values():
        spine.set_visible(False)
    for i in range(2):
        for j in range(2):
            value = matrix[i][j]
            text_color = "#ffffff" if value > vmax * 0.55 else INK_PRIMARY
            ax_cm.text(j, i, str(value), ha="center", va="center",
                       color=text_color, fontsize=16, fontweight="bold")

    # --- Hit rate / false-alarm rate bars ---
    labels = ["Hit rate\n(probe = target)", "False-alarm rate\n(probe = distractor)"]
    rates = [d["hit_rate"], d["false_alarm_rate"]]
    colors = [CATEGORICAL_1, CATEGORICAL_2]
    bars = ax_rates.bar(labels, rates, color=colors, width=0.55, zorder=3)
    ax_rates.axhline(0.5, color=INK_MUTED, linewidth=1, linestyle="--", zorder=2)
    ax_rates.set_ylim(0, 1)
    ax_rates.set_ylabel("Rate", color=INK_SECONDARY)
    ax_rates.tick_params(colors=INK_SECONDARY)
    ax_rates.spines["top"].set_visible(False)
    ax_rates.spines["right"].set_visible(False)
    ax_rates.spines["left"].set_color(GRIDLINE)
    ax_rates.spines["bottom"].set_color(GRIDLINE)
    ax_rates.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
    for bar, rate in zip(bars, rates):
        ax_rates.text(bar.get_x() + bar.get_width() / 2, rate + 0.02, f"{rate:.2f}",
                       ha="center", va="bottom", color=INK_PRIMARY, fontsize=10)
    ax_rates.set_title(
        f"d' = {d['d_prime']:.2f}   (criterion c = {d['criterion']:.2f})",
        color=INK_PRIMARY, fontsize=11, pad=10,
    )

    footer = (
        f"n_target={d['n_target_trials']}  n_distractor={d['n_distractor_trials']}"
        f"  n_unparseable={d['n_unparseable']}"
    )
    fig.text(0.5, 0.01, footer, ha="center", color=INK_MUTED, fontsize=9)

    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    print(f"wrote {out_path}")
    print(d)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_csv", type=str)
    parser.add_argument("--title", type=str, default="")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    out_path = args.out or args.results_csv.replace("results.csv", "confusion_and_dprime.png")
    plot(args.results_csv, args.title, out_path)


if __name__ == "__main__":
    main()
