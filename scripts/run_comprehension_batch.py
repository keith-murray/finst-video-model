"""
Batch runner for the comprehension arm: sweeps n_circles x seeds x models,
running the same generate_stimulus -> ask_about_video -> score_answer
pipeline as run_comprehension_trial.py for every combination, and collects
every trial's result as one row in a single results.csv.

Unlike run_comprehension_trial.py (one trial per invocation, its own
data/<trial_id>/ directory), every trial launched by one invocation of this
script shares a single batch_id, split across two gitignore-aware locations:
individual trial artifacts (video, ground truth, question, result -- large,
regeneratable) live under data/<batch_id>/trials/<trial_id>/ (data/ is
gitignored), while config.json and results.csv (small, the actual analysis
output) go to results/<batch_id>/ so they're tracked in git.

The question-building logic is imported from run_comprehension_trial.py
rather than duplicated, since both scripts ask the exact same
comprehension-arm question -- only the model/seed/n_circles grid differs
here.

Usage:
    uv run python scripts/run_comprehension_batch.py
    uv run python scripts/run_comprehension_batch.py \\
        --n-circles 2 4 6 8 10 --seeds 0 1 2 \\
        --models google/gemini-2.5-flash qwen/qwen3.5-397b-a17b
"""

import argparse
import csv
import json
import os
import uuid

import requests

from finst_video_model.comprehension.config import TrialConfig
from finst_video_model.comprehension.scoring import parse_answer_letters, score_answer
from finst_video_model.comprehension.stimulus_gen import generate_stimulus
from finst_video_model.comprehension.vlm_client import ask_about_video
from run_comprehension_trial import build_question

DEFAULT_MODELS = ["google/gemini-2.5-flash"]
DEFAULT_N_CIRCLES = [3, 4, 6, 8, 10]
DEFAULT_SEEDS = [0, 1, 2]

RESULT_FIELDS = [
    "trial_id", "model", "seed", "n_circles", "n_cued",
    "cued_letters", "predicted_letters", "exact_match", "correct_count",
    "precision", "recall", "error",
]


def run_one_trial(batch_dir: str, model: str, n_circles: int, n_cued: int, seed: int) -> dict:
    """Runs one trial into data/<batch_id>/trials/<trial_id>/ and returns its
    results.csv row (as a dict)."""
    cfg = TrialConfig(n_circles=n_circles, n_cued=n_cued, seed=seed)
    trial_dir = os.path.join(batch_dir, "trials", cfg.trial_id)
    os.makedirs(trial_dir, exist_ok=True)

    ground_truth = generate_stimulus(cfg, trial_dir)
    question = build_question(cfg)
    with open(os.path.join(trial_dir, "question.txt"), "w") as f:
        f.write(question)

    result = {
        "trial_id": cfg.trial_id,
        "model": model,
        "seed": seed,
        "video_path": ground_truth["video_path"],
        "question": question,
        "cued_letters": ground_truth["cued_letters"],
    }
    row = {
        "trial_id": cfg.trial_id,
        "model": model,
        "seed": seed,
        "n_circles": n_circles,
        "n_cued": n_cued,
        "cued_letters": ",".join(ground_truth["cued_letters"]),
        "predicted_letters": "",
        "exact_match": "",
        "correct_count": "",
        "precision": "",
        "recall": "",
        "error": "",
    }

    try:
        response_text = ask_about_video(ground_truth["video_path"], question, model)
        predicted_letters = parse_answer_letters(response_text)
        score = score_answer(predicted_letters, ground_truth["cued_letters"])
        result["response"] = response_text
        result["score"] = score
        row.update({
            "predicted_letters": ",".join(score["predicted_letters"]),
            "exact_match": score["exact_match"],
            "correct_count": score["correct_count"],
            "precision": score["precision"],
            "recall": score["recall"],
        })
    except (requests.RequestException, KeyError, RuntimeError) as e:
        result["error"] = str(e)
        row["error"] = str(e)

    with open(os.path.join(trial_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)

    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-circles", type=int, nargs="+", default=DEFAULT_N_CIRCLES)
    parser.add_argument("--n-cued", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--models", type=str, nargs="+", default=DEFAULT_MODELS)
    args = parser.parse_args()

    batch_id = uuid.uuid4().hex[:8]
    batch_dir = os.path.join("data", batch_id)
    results_dir = os.path.join("results", batch_id)
    os.makedirs(os.path.join(batch_dir, "trials"), exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, "config.json"), "w") as f:
        json.dump({
            "batch_id": batch_id,
            "n_circles": args.n_circles,
            "n_cued": args.n_cued,
            "seeds": args.seeds,
            "models": args.models,
        }, f, indent=2)

    total = len(args.n_circles) * len(args.seeds) * len(args.models)
    rows = []
    done = 0
    for n_circles in args.n_circles:
        for seed in args.seeds:
            for model in args.models:
                done += 1
                print(f"[{done}/{total}] model={model} n_circles={n_circles} seed={seed}")
                rows.append(run_one_trial(batch_dir, model, n_circles, args.n_cued, seed))

    results_csv = os.path.join(results_dir, "results.csv")
    with open(results_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} trial results to {results_csv}")
    print(f"Raw trial artifacts (video/ground_truth/response) in {batch_dir}/trials/")


if __name__ == "__main__":
    main()
