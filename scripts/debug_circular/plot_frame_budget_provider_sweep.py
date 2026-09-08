"""
Summary figure for the per-provider frame-budget sweep
(results/debug_circular/frame_budget_provider_sweep/results.csv, produced by
run_gemma_frame_budget_provider_sweep.py -- see reference_gemma_fixed_frame_
budget memory's "Confirmed provider-independent" section): one line per
OpenRouter provider, `prompt_tokens` vs. real frame count (1-45), pinned via
`provider={"only": [p], "allow_fallbacks": False}`.

Six of seven providers (deepinfra, coreweave, crusoe, parasail, together,
modelrun) land almost exactly on top of each other -- the point of this
figure is to show that visually. `chutes` is plotted as a dashed line with
gaps where sustained 503 errors (n>=15, most points) left no data, rather
than interpolated over.

Colors: 6 categorical hues (validated slots) for the reliable providers,
distinguished by line style/marker only since values overlap almost
perfectly; chutes gets a 7th, dashed and less prominent given its data gaps.

Usage:
    uv run python scripts/debug_circular/plot_frame_budget_provider_sweep.py
"""

import csv
from collections import defaultdict

import matplotlib.pyplot as plt

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

RESULTS_CSV = "results/debug_circular/frame_budget_provider_sweep/results.csv"
OUT_PATH = "results/debug_circular/frame_budget_provider_sweep/prompt_tokens_by_provider.png"

# 6 validated categorical hues + 1 muted extra for the unreliable provider
PROVIDER_COLORS = {
    "deepinfra": "#2a78d6",
    "coreweave": "#eb6834",
    "crusoe": "#2fa88a",
    "parasail": "#9558c7",
    "together": "#c2b32a",
    "modelrun": "#d64f7a",
    "chutes": "#b6b4ac",
}
PROVIDER_ORDER = ["deepinfra", "coreweave", "crusoe", "parasail", "together", "modelrun", "chutes"]


def tokens_by_provider(results_csv: str) -> dict[str, dict[int, int]]:
    out = defaultdict(dict)
    with open(results_csv) as f:
        for row in csv.DictReader(f):
            if row["prompt_tokens"]:
                out[row["provider_pinned"]][int(row["n_frames"])] = int(row["prompt_tokens"])
    return out


def main():
    by_provider = tokens_by_provider(RESULTS_CSV)

    fig, ax = plt.subplots(figsize=(9, 6), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, axis="y", color=GRIDLINE, linewidth=1, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED)

    for provider in PROVIDER_ORDER:
        if provider not in by_provider:
            continue
        data = by_provider[provider]
        ns = sorted(data)
        is_chutes = provider == "chutes"
        ax.plot(
            ns, [data[n] for n in ns],
            color=PROVIDER_COLORS[provider], label=provider,
            linewidth=2 if not is_chutes else 1.5,
            linestyle="--" if is_chutes else "-",
            marker="o", markersize=3.5, alpha=1.0 if not is_chutes else 0.9,
            zorder=4 if not is_chutes else 5,
        )

    ax.set_xlabel("Real frame count", color=INK_SECONDARY)
    ax.set_ylabel("prompt_tokens", color=INK_SECONDARY)
    ax.set_title(
        "google/gemma-4-31b-it: frame-budget sweep by OpenRouter provider",
        color=INK_PRIMARY, fontsize=13, fontweight="bold",
    )
    fig.suptitle(
        "(6/7 providers overlap almost exactly at 23 + 73*min(n,32) tokens;\n"
        "chutes dashed -- sustained 503s left gaps at n>=15)",
        color=INK_MUTED, fontsize=10, y=0.94,
    )
    ax.legend(frameon=False, labelcolor=INK_PRIMARY, fontsize=9, loc="lower right")

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
