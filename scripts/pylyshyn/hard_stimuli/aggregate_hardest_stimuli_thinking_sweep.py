"""
Aggregates `run_hardest_stimuli_thinking_sweep.py`'s results.csv into a
per-model accuracy/d'/cost summary, per `claude/2026_09/2026_09_16/
TODO.md`'s Part 4 (reasoning="medium" pass). Direct mirror of
`aggregate_hardest_stimuli_sweep.py` (no-reasoning pass) pointed at the
reasoning="medium" results directory instead.

Writes results/pylyshyn/hard_stimuli/hardest_stimuli_thinking_sweep/summary.json.

Usage:
    uv run python scripts/pylyshyn/hard_stimuli/aggregate_hardest_stimuli_thinking_sweep.py
"""

import csv
import json
import math
import os

from finst_video_model.scoring import compute_d_prime
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nearest_neighbor_heuristic import heuristic_accuracy_by_group

RESULTS_CSV = "results/pylyshyn/hard_stimuli/hardest_stimuli_thinking_sweep/results.csv"
TRIALS_ROOT = "data/pylyshyn/hardest_stimuli_sweep/trials"
OUT_PATH = "results/pylyshyn/hard_stimuli/hardest_stimuli_thinking_sweep/summary.json"


def accuracy_and_sem(rows: list[dict]) -> tuple[float, float, int]:
    n = len(rows)
    correct = sum(1 for r in rows if r["predicted"] == r["probe_is_target"])
    p = correct / n
    sem = math.sqrt(p * (1 - p) / n) * 100
    return p * 100, sem, n


def main():
    with open(RESULTS_CSV) as f:
        all_rows = list(csv.DictReader(f))

    models = sorted(set(r["model"] for r in all_rows))
    summary = {}

    for model in models:
        model_rows = [r for r in all_rows if r["model"] == model]
        scorable = [
            {
                "probe_is_target": r["probe_is_target"] == "True",
                "predicted": None if r["predicted"] == "" else r["predicted"] == "True",
            }
            for r in model_rows
        ]
        acc, sem, n = accuracy_and_sem([
            {"probe_is_target": s["probe_is_target"], "predicted": s["predicted"]}
            for s in scorable if s["predicted"] is not None
        ])
        d_prime_stats = compute_d_prime(scorable)
        total_cost = sum(float(r["cost"]) for r in model_rows if r["cost"])
        total_reasoning_tokens = sum(int(r["reasoning_tokens"]) for r in model_rows if r["reasoning_tokens"])
        n_errors = sum(1 for r in model_rows if r["error"])

        summary[model] = {
            "n_trials": len(model_rows),
            "n_errors": n_errors,
            "n_unparseable": d_prime_stats["n_unparseable"],
            "accuracy_pct": acc,
            "accuracy_sem_pct": sem,
            "d_prime": d_prime_stats["d_prime"],
            "criterion": d_prime_stats["criterion"],
            "total_cost_usd": total_cost,
            "total_reasoning_tokens": total_reasoning_tokens,
            "mean_reasoning_tokens": total_reasoning_tokens / len(model_rows) if model_rows else 0,
        }
        print(
            f"model={model}: acc={acc:.1f}% (sem={sem:.1f}%, n={n}), "
            f"d'={d_prime_stats['d_prime']:.2f}, errors={n_errors}, "
            f"unparseable={d_prime_stats['n_unparseable']}, cost=${total_cost:.4f}, "
            f"mean_reasoning_tokens={summary[model]['mean_reasoning_tokens']:.0f}"
        )

    heuristic_stats = heuristic_accuracy_by_group(all_rows, TRIALS_ROOT, lambda row: "n3")
    h_acc, h_sem, h_n = heuristic_stats["n3"]
    summary["_heuristic"] = {"accuracy_pct": h_acc, "accuracy_sem_pct": h_sem, "n_trials": h_n}
    print(f"\nnearest-neighbor heuristic (reverified): acc={h_acc:.1f}% (sem={h_sem:.1f}%, n={h_n})")

    summary["_totals"] = {
        "total_cost_usd": sum(v["total_cost_usd"] for k, v in summary.items() if not k.startswith("_")),
    }
    print(f"\nTotal actual cost across all models: ${summary['_totals']['total_cost_usd']:.4f}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
