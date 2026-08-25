"""
Aggregates raw cluster_response.json outputs (written by
scripts/debug_circular/run_cluster_batch.py on the cluster, then rsynced
back down) into a scored results.csv, reusing this project's existing
True/False parsing and signal-detection scoring
(finst_video_model.scoring) rather than reimplementing it in the cluster's
standalone script.

Unlike scripts/pylyshyn/run_pylyshyn_batch.py's incremental-append
results.csv (needed there to avoid re-paying for OpenRouter API calls on a
partial rerun), this script fully regenerates results.csv from scratch
every run -- aggregation here is a cheap local JSON-parsing pass with
nothing to avoid re-doing, so "regenerate fresh" sidesteps any risk of
stale rows if a trial's raw output is corrected/re-pulled.

Usage:
    uv run python scripts/debug_circular/aggregate_cluster_results.py --run-name <run_name>
"""

import argparse
import csv
import json
import os

from finst_video_model.scoring import classify_trial, compute_d_prime, parse_boolean_answer

RESULT_FIELDS = [
    "trial_id", "model_tag", "n_objects", "rotation_deg", "clockwise",
    "probe_on_target", "seed", "probe_is_target", "response", "predicted",
    "outcome", "prompt_tokens", "output_tokens", "generation_time_s", "error",
]


def load_trial_row(trial_dir: str, trial_id: str) -> dict | None:
    gt_path = os.path.join(trial_dir, "ground_truth.json")
    response_path = os.path.join(trial_dir, "cluster_response.json")
    if not os.path.isfile(gt_path):
        print(f"[{trial_id}] missing ground_truth.json, skipping")
        return None
    if not os.path.isfile(response_path):
        print(f"[{trial_id}] no cluster_response.json yet, skipping")
        return None

    with open(gt_path) as f:
        ground_truth = json.load(f)
    with open(response_path) as f:
        response = json.load(f)

    cfg = ground_truth["config"]
    response_text = response.get("response_text")
    predicted = parse_boolean_answer(response_text) if response_text else None
    if predicted is not None:
        outcome = classify_trial(ground_truth["probe_is_target"], predicted)
    else:
        outcome = "unparseable" if response_text else ""

    return {
        "trial_id": trial_id,
        "model_tag": response.get("model_tag"),
        "n_objects": cfg["n_objects"],
        "rotation_deg": cfg["rotation_deg"],
        "clockwise": cfg["clockwise"],
        "probe_on_target": cfg["probe_on_target"],
        "seed": cfg["seed"],
        "probe_is_target": ground_truth["probe_is_target"],
        "response": response_text,
        "predicted": predicted,
        "outcome": outcome,
        "prompt_tokens": response.get("prompt_tokens"),
        "output_tokens": response.get("output_tokens"),
        "generation_time_s": response.get("generation_time_s"),
        "error": response.get("error"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument(
        "--trials-root", type=str, default=None,
        help="Defaults to data/debug_circular/<run-name>/trials",
    )
    parser.add_argument(
        "--out-root", type=str, default=None,
        help="Defaults to results/debug_circular/<run-name>",
    )
    args = parser.parse_args()

    trials_root = args.trials_root or os.path.join("data", "debug_circular", args.run_name, "trials")
    out_root = args.out_root or os.path.join("results", "debug_circular", args.run_name)
    os.makedirs(out_root, exist_ok=True)

    if not os.path.isdir(trials_root):
        raise SystemExit(f"trials-root {trials_root} does not exist")

    trial_ids = sorted(
        d for d in os.listdir(trials_root)
        if os.path.isdir(os.path.join(trials_root, d))
    )
    rows = [
        row for row in (
            load_trial_row(os.path.join(trials_root, trial_id), trial_id)
            for trial_id in trial_ids
        )
        if row is not None
    ]

    n_found = len(rows)
    n_missing = len(trial_ids) - n_found
    print(f"{n_found} trials scored, {n_missing} skipped (missing ground truth or response)")

    results_csv = os.path.join(out_root, "results.csv")
    with open(results_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {results_csv}")

    config_path = os.path.join(out_root, "config.json")
    with open(config_path, "w") as f:
        json.dump({
            "run_name": args.run_name,
            "arm": "debug_circular",
            "trials_root": trials_root,
            "n_trials_found": n_found,
            "n_missing": n_missing,
        }, f, indent=2)
    print(f"Wrote {config_path}")

    scorable = [r for r in rows if r["predicted"] is not None]
    print(f"\n--- d' by rotation_deg (n={len(scorable)} scorable trials) ---")
    by_rotation = {}
    for row in scorable:
        by_rotation.setdefault(row["rotation_deg"], []).append(row)
    for rotation_deg in sorted(by_rotation):
        stats = compute_d_prime(by_rotation[rotation_deg])
        print(
            f"rotation_deg={rotation_deg}: d'={stats['d_prime']:.2f} "
            f"(hits={stats['hits']}, misses={stats['misses']}, "
            f"false_alarms={stats['false_alarms']}, "
            f"correct_rejections={stats['correct_rejections']}, "
            f"unparseable={stats['n_unparseable']})"
        )


if __name__ == "__main__":
    main()
