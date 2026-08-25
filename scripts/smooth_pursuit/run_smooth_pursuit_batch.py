"""
Batch runner for the smooth_pursuit arm's "1/5 cued" experiment
(claude/2026_08_14/TODO.md's "Let's run some simple experiments" section):
n_circles=5, n_cued=1, 20 seeds, each seed run twice -- once with
probe_on_target=True, once False -- so the two runs share an identical
underlying scene (placement/cueing/motion all derive from `seed` alone in
smooth_pursuit.physics.build_circles; only which circle gets probed
differs), giving a paired 40-trial design: 20 target-probe trials for the
hit rate, 20 distractor-probe trials for the false-alarm rate, as needed by
finst_video_model.scoring.compute_d_prime.

Model is pinned to google/gemini-3.7-flash on the google-vertex provider
specifically (half the price of the default google-ai-studio provider, per
OpenRouter's /models/{id}/endpoints), with reasoning effort capped at
"minimal" -- the cheapest allowed setting; this endpoint 400s if reasoning
is disabled outright ("reasoning is mandatory for this endpoint"), and even
"minimal" isn't a hard token cap (Gemini's actual reasoning-token spend is
decided internally by Google regardless of the requested effort level, per
OpenRouter's docs). Measured cost during design: ~$0.0026-0.0028/trial.

Every trial's artifacts land in data/<batch_id>/trials/<trial_id>/ (video,
ground_truth, question, result -- data/ is gitignored); results.csv and
config.json (small, tracked) go to results/<batch_id>/, matching the
archived scripts/prototype/run_comprehension_batch.py convention.
results.csv is written incrementally so an interrupted run can be resumed
with --resume <batch_id> (already-completed (seed, probe_on_target) pairs
are skipped, not re-run/re-paid-for).

Usage:
    uv run python scripts/smooth_pursuit/run_smooth_pursuit_batch.py
    uv run python scripts/smooth_pursuit/run_smooth_pursuit_batch.py --resume <batch_id>
"""

import argparse
import concurrent.futures
import csv
import json
import os
import uuid

import requests

from finst_video_model.smooth_pursuit.config import TrialConfig
from finst_video_model.smooth_pursuit.stimulus_gen import generate_stimulus
from finst_video_model.scoring import classify_trial, parse_boolean_answer
from finst_video_model.vlm_client import ask_about_video
from run_smooth_pursuit_trial import build_question

MODEL = "google/gemini-3.7-flash"
PROVIDER = {"only": ["google-vertex"], "allow_fallbacks": False}
REASONING = {"effort": "minimal"}
N_CIRCLES = 5
N_CUED = 1
N_SEEDS = 20

RESULT_FIELDS = [
    "trial_id", "seed", "n_circles", "n_cued", "probe_on_target", "probe_is_target",
    "response", "predicted", "outcome", "cost", "error",
]


def run_one_trial(batch_dir: str, seed: int, probe_on_target: bool) -> dict:
    cfg = TrialConfig(n_circles=N_CIRCLES, n_cued=N_CUED, probe_on_target=probe_on_target, seed=seed)
    trial_dir = os.path.join(batch_dir, "trials", cfg.trial_id)
    os.makedirs(trial_dir, exist_ok=True)

    ground_truth = generate_stimulus(cfg, trial_dir)
    question = build_question(cfg)
    with open(os.path.join(trial_dir, "question.txt"), "w") as f:
        f.write(question)

    result = {
        "trial_id": cfg.trial_id, "seed": seed, "n_circles": N_CIRCLES, "n_cued": N_CUED,
        "probe_on_target": probe_on_target,
        "probe_is_target": ground_truth["probe_is_target"],
        "video_path": ground_truth["video_path"], "question": question,
    }
    row = {
        "trial_id": cfg.trial_id, "seed": seed, "n_circles": N_CIRCLES, "n_cued": N_CUED,
        "probe_on_target": probe_on_target,
        "probe_is_target": ground_truth["probe_is_target"],
        "response": "", "predicted": "", "outcome": "", "cost": "", "error": "",
    }

    try:
        response_text, usage = ask_about_video(
            ground_truth["video_path"], question, MODEL,
            provider=PROVIDER, reasoning=REASONING, return_usage=True,
        )
        predicted = parse_boolean_answer(response_text)
        outcome = (
            classify_trial(ground_truth["probe_is_target"], predicted)
            if predicted is not None else "unparseable"
        )
        result.update({"response": response_text, "predicted": predicted, "outcome": outcome, "usage": usage})
        row.update({
            "response": response_text, "predicted": predicted, "outcome": outcome,
            "cost": usage.get("cost") if usage else "",
        })
    except (requests.RequestException, KeyError, RuntimeError) as e:
        result["error"] = str(e)
        row["error"] = str(e)

    with open(os.path.join(trial_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)

    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(N_SEEDS)))
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--concurrency", type=int, default=5)
    args = parser.parse_args()

    if args.resume:
        batch_id = args.resume
    else:
        batch_id = uuid.uuid4().hex[:8]
    batch_dir = os.path.join("data", batch_id)
    results_dir = os.path.join("results", batch_id)
    os.makedirs(os.path.join(batch_dir, "trials"), exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    config_path = os.path.join(results_dir, "config.json")
    if not os.path.isfile(config_path):
        with open(config_path, "w") as f:
            json.dump({
                "batch_id": batch_id, "arm": "smooth_pursuit", "model": MODEL,
                "provider": PROVIDER, "reasoning": REASONING,
                "n_circles": N_CIRCLES, "n_cued": N_CUED, "seeds": args.seeds,
            }, f, indent=2)

    results_csv = os.path.join(results_dir, "results.csv")
    already_ran = set()
    file_exists = os.path.isfile(results_csv)
    if file_exists:
        with open(results_csv) as f:
            for r in csv.DictReader(f):
                already_ran.add((int(r["seed"]), r["probe_on_target"] == "True"))

    pending = [
        (seed, probe_on_target)
        for seed in args.seeds
        for probe_on_target in (True, False)
        if (seed, probe_on_target) not in already_ran
    ]
    total = len(args.seeds) * 2
    done = len(already_ran)

    with open(results_csv, "a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=RESULT_FIELDS)
        if not file_exists:
            writer.writeheader()
            csv_file.flush()

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {
                pool.submit(run_one_trial, batch_dir, seed, probe_on_target): (seed, probe_on_target)
                for seed, probe_on_target in pending
            }
            for future in concurrent.futures.as_completed(futures):
                seed, probe_on_target = futures[future]
                row = future.result()
                done += 1
                print(f"[{done}/{total}] seed={seed} probe_on_target={probe_on_target} -> {row['outcome'] or row['error']}")
                writer.writerow(row)
                csv_file.flush()

    print(f"\nresults.csv up to date at {results_csv}")
    print(f"batch_id={batch_id}")


if __name__ == "__main__":
    main()
