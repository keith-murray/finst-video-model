"""
Aggregates raw cluster_response.json outputs (written by
scripts/pylyshyn/gemma_local/run_gemma_local_batch.py on the cluster, then rsynced back
down) into a scored results.csv, reusing this project's existing True/False
parsing and signal-detection scoring (finst_video_model.scoring) -- same
approach as scripts/debug_circular/aggregate_gemma_frame_sweep.py.

Every trial here shares one fixed condition (redirect=fast, speed=fast,
variant=native, num_frames=32, sampling="recommended") -- only n_objects
and seed vary -- so those constants are hardcoded rather than swept, unlike
the debug_circular aggregator's num_frames/sampling axes.

Usage:
    uv run python scripts/pylyshyn/gemma_local/aggregate_gemma_local_nobjects_sweep.py
"""

import argparse
import csv
import json
import os

from finst_video_model.scoring import classify_trial, compute_d_prime, parse_boolean_answer

MODEL_TAG = "google/gemma-4-31b-it-local"
REDIRECT_CONDITION = "fast"
SPEED_CONDITION = "fast"
VARIANT = "native"
NUM_FRAMES = 32
SAMPLING = "recommended"

RESULT_FIELDS = [
    "trial_id", "model", "n_objects", "redirect_condition", "speed_condition", "variant",
    "seed", "probe_on_target", "probe_is_target", "response", "predicted", "outcome",
    "prompt_tokens", "output_tokens", "generation_time_s", "error",
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
    if isinstance(response_text, dict):
        response_text = response_text.get("content")
    predicted = parse_boolean_answer(response_text) if response_text else None
    if predicted is not None:
        outcome = classify_trial(ground_truth["probe_is_target"], predicted)
    else:
        outcome = "unparseable" if response_text else ""

    return {
        "trial_id": trial_id,
        "model": MODEL_TAG,
        "n_objects": cfg["n_objects"],
        "redirect_condition": REDIRECT_CONDITION,
        "speed_condition": SPEED_CONDITION,
        "variant": VARIANT,
        "seed": cfg["seed"],
        "probe_on_target": cfg["probe_on_target"],
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
    parser.add_argument(
        "--trials-root", type=str,
        default="data/pylyshyn/gemma_local_nobjects_sweep/trials",
    )
    parser.add_argument(
        "--out-root", type=str, default="results/pylyshyn/gemma_local/gemma_local_nobjects_sweep",
    )
    args = parser.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    if not os.path.isdir(args.trials_root):
        raise SystemExit(f"trials-root {args.trials_root} does not exist")

    trial_ids = sorted(
        d for d in os.listdir(args.trials_root)
        if os.path.isdir(os.path.join(args.trials_root, d))
    )

    rows = []
    for trial_id in trial_ids:
        row = load_trial_row(os.path.join(args.trials_root, trial_id), trial_id)
        if row is not None:
            rows.append(row)

    print(f"{len(rows)}/{len(trial_ids)} trials scored")

    results_csv = os.path.join(args.out_root, "results.csv")
    with open(results_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {results_csv}")

    config_path = os.path.join(args.out_root, "config.json")
    with open(config_path, "w") as f:
        json.dump({
            "run_name": "gemma_local_nobjects_sweep",
            "arm": "pylyshyn",
            "model": MODEL_TAG,
            "trials_root": args.trials_root,
            "redirect_condition": REDIRECT_CONDITION,
            "speed_condition": SPEED_CONDITION,
            "variant": VARIANT,
            "num_frames": NUM_FRAMES,
            "sampling": SAMPLING,
            "n_trials_found": len(rows),
            "n_trials_expected": len(trial_ids),
        }, f, indent=2)
    print(f"Wrote {config_path}")

    scorable = [r for r in rows if r["predicted"] is not None]
    print(f"\n--- d' by n_objects (n={len(scorable)} scorable trials) ---")
    by_n_objects = {}
    for row in scorable:
        by_n_objects.setdefault(row["n_objects"], []).append(row)
    for n_objects in sorted(by_n_objects):
        stats = compute_d_prime(by_n_objects[n_objects])
        print(
            f"n_objects={n_objects}: d'={stats['d_prime']:.2f} "
            f"(hits={stats['hits']}, misses={stats['misses']}, "
            f"false_alarms={stats['false_alarms']}, "
            f"correct_rejections={stats['correct_rejections']}, "
            f"unparseable={stats['n_unparseable']})"
        )


if __name__ == "__main__":
    main()
