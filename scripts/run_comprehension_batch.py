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

results.csv is written incrementally (one row appended per completed trial,
flushed to disk immediately), not buffered in memory until the end -- a
sweep killed partway through (background task limits, network drop, etc.)
still leaves every completed trial's row on disk. Pass --resume <batch_id>
to continue an interrupted or extended sweep: already-completed
model/n_circles/seed combinations (read back from the existing results.csv)
are skipped rather than re-run and re-paid-for, and the original run's grid
and timing come from its saved config.json rather than being re-specified.

Trials run concurrently, up to --concurrency at a time (default 5), via a
thread pool: each trial's ask_about_video call is latency-bound (waiting on
OpenRouter, not local CPU), so running several in flight at once cuts
wall-clock time roughly proportionally. Stimulus generation and the VLM
client use only local, per-call RNG/file state (no shared globals), so
trials don't interfere with each other; results.csv is still only ever
written from the main thread as each trial completes, so no locking is
needed there. Progress printouts and the --limit cutoff are in
completion order, not the original sweep order, when concurrency > 1.

Usage:
    uv run python scripts/run_comprehension_batch.py
    uv run python scripts/run_comprehension_batch.py \\
        --n-circles 2 4 6 8 10 --seeds 0 1 2 \\
        --models google/gemini-2.5-flash qwen/qwen3.5-397b-a17b \\
        --cue-flash-s 2 --tracking-s 6 --label-s 2
    uv run python scripts/run_comprehension_batch.py --resume ce675830 --concurrency 10
"""

import argparse
import concurrent.futures
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


def run_one_trial(
    batch_dir: str, model: str, n_circles: int, n_cued: int, seed: int,
    cue_flash_s: float, tracking_s: float, label_s: float,
) -> dict:
    """Runs one trial into data/<batch_id>/trials/<trial_id>/ and returns its
    results.csv row (as a dict)."""
    cfg = TrialConfig(
        n_circles=n_circles, n_cued=n_cued, seed=seed,
        cue_flash_s=cue_flash_s, tracking_s=tracking_s, label_s=label_s,
    )
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
    parser.add_argument("--cue-flash-s", type=float, default=1.0)
    parser.add_argument("--tracking-s", type=float, default=8.0)
    parser.add_argument("--label-s", type=float, default=2.0)
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Existing batch_id to continue -- reuses its saved config.json "
             "grid/timing and skips model/n_circles/seed combos already in "
             "its results.csv",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Stop after running this many new trials in this invocation "
             "(e.g. to stay under a background-task duration limit); "
             "resume with --resume <batch_id> to continue",
    )
    parser.add_argument(
        "--concurrency", type=int, default=5,
        help="Number of trials to run at once via a thread pool (each trial's "
             "OpenRouter call is latency-bound, so this cuts wall-clock time "
             "roughly proportionally). Safe to raise above OpenRouter's 20 "
             "requests/minute limit -- vlm_client.ask_about_video enforces "
             "that cap itself across all threads, so --concurrency only "
             "controls local parallelism (stimulus rendering etc.), not the "
             "actual request rate",
    )
    args = parser.parse_args()

    batch_dir_for = lambda bid: os.path.join("data", bid)
    results_dir_for = lambda bid: os.path.join("results", bid)

    if args.resume:
        batch_id = args.resume
        batch_dir = batch_dir_for(batch_id)
        results_dir = results_dir_for(batch_id)
        config_path = os.path.join(results_dir, "config.json")
        if not os.path.isfile(config_path):
            raise SystemExit(f"--resume: {config_path} not found")
        with open(config_path) as f:
            cfg_saved = json.load(f)
        n_circles_list = cfg_saved["n_circles"]
        n_cued = cfg_saved["n_cued"]
        seeds_list = cfg_saved["seeds"]
        models_list = cfg_saved["models"]
        cue_flash_s = cfg_saved["cue_flash_s"]
        tracking_s = cfg_saved["tracking_s"]
        label_s = cfg_saved["label_s"]
    else:
        batch_id = uuid.uuid4().hex[:8]
        batch_dir = batch_dir_for(batch_id)
        results_dir = results_dir_for(batch_id)
        os.makedirs(os.path.join(batch_dir, "trials"), exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        n_circles_list = args.n_circles
        n_cued = args.n_cued
        seeds_list = args.seeds
        models_list = args.models
        cue_flash_s = args.cue_flash_s
        tracking_s = args.tracking_s
        label_s = args.label_s
        with open(os.path.join(results_dir, "config.json"), "w") as f:
            json.dump({
                "batch_id": batch_id,
                "n_circles": n_circles_list,
                "n_cued": n_cued,
                "seeds": seeds_list,
                "models": models_list,
                "cue_flash_s": cue_flash_s,
                "tracking_s": tracking_s,
                "label_s": label_s,
            }, f, indent=2)

    results_csv = os.path.join(results_dir, "results.csv")
    already_ran = set()
    file_exists = os.path.isfile(results_csv)
    if file_exists:
        with open(results_csv) as f:
            for r in csv.DictReader(f):
                already_ran.add((r["model"], int(r["n_circles"]), int(r["seed"])))

    pending = [
        (model, n_circles, seed)
        for n_circles in n_circles_list
        for seed in seeds_list
        for model in models_list
        if (model, n_circles, seed) not in already_ran
    ]
    remaining_after_limit = 0
    if args.limit is not None and len(pending) > args.limit:
        remaining_after_limit = len(pending) - args.limit
        pending = pending[: args.limit]

    total = len(n_circles_list) * len(seeds_list) * len(models_list)
    done = len(already_ran)
    with open(results_csv, "a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=RESULT_FIELDS)
        if not file_exists:
            writer.writeheader()
            csv_file.flush()

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {
                pool.submit(
                    run_one_trial, batch_dir, model, n_circles, n_cued, seed,
                    cue_flash_s, tracking_s, label_s,
                ): (model, n_circles, seed)
                for model, n_circles, seed in pending
            }
            for future in concurrent.futures.as_completed(futures):
                model, n_circles, seed = futures[future]
                row = future.result()
                done += 1
                print(f"[{done}/{total}] model={model} n_circles={n_circles} seed={seed}")
                writer.writerow(row)
                csv_file.flush()

    if remaining_after_limit:
        print(
            f"\nReached --limit {args.limit} new trials; stopping early. "
            f"Resume with --resume {batch_id} to continue "
            f"({remaining_after_limit} trial(s) remaining)."
        )

    print(f"\nresults.csv up to date at {results_csv}")
    print(f"Raw trial artifacts (video/ground_truth/response) in {batch_dir}/trials/")


if __name__ == "__main__":
    main()
