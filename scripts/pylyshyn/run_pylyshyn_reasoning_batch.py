"""
Batch runner for claude/2026_08_20/TODO.md's reasoning-budget sweep: does
Aug-19's finding that reasoning effort matters more than model choice
(claude/2026_08_19/SUMMARY.md -- qwen3.8-27b at "low" beat both its own
disabled-reasoning run and qwen3.8-max at its mandatory "minimal" floor)
hold across the *full* range of reasoning effort, for both models?

Fixed at the simplest pylyshyn condition (n_objects=2, n_cued=1) per the
TODO, sweeping only OpenRouter's `reasoning.effort` levels instead. Tried
`reasoning.max_tokens` as a literal numeric "budget" first (see the
run_pylyshyn_reasoning_batch git commit message / session notes) but a
same-question pilot showed it isn't actually enforced as a hard cap --
budget=100 still spent 654 reasoning tokens on qwen3.8-27b -- so it can't
serve as a controlled independent variable. `effort` is the knob OpenRouter
actually exposes as a requestable "budget" tier (and is respected: a pilot
on qwen3.8-max showed reasoning_tokens roughly 5x higher at
medium/high vs. minimal/low), so REASONING_LEVELS orders it low-to-high and
that order is the sweep's x-axis. qwen3.8-27b's reasoning is optional, so
it additionally gets a "none" (`{"enabled": False}`) point at the bottom of
its own range; qwen3.8-max's is mandatory (confirmed live: disabling 400s
with "Reasoning is mandatory for this endpoint and cannot be disabled"), so
its sweep starts at "minimal".

Each (model, reasoning_level) condition gets N_SEEDS seeds split evenly
(first half probed on-target, second half on-distractor), same
matching/non-matching convention as run_pylyshyn_batch.py, needed for
compute_d_prime / percent-error scoring.

Every trial's artifacts land in data/pylyshyn/reasoning_sweep/trials/<trial_id>/
(gitignored); results.csv/config.json (tracked) go to
results/pylyshyn/reasoning_sweep/. results.csv is written incrementally and
re-running the same --models/--seeds resumes automatically (already-run
(model, reasoning_level, seed, probe_on_target) combinations are skipped).

Usage:
    uv run python scripts/pylyshyn/run_pylyshyn_reasoning_batch.py
    uv run python scripts/pylyshyn/run_pylyshyn_reasoning_batch.py --models qwen/qwen3.8-27b
"""

import argparse
import concurrent.futures
import csv
import json
import os

import requests

from finst_video_model.pylyshyn.config import TrialConfig
from finst_video_model.pylyshyn.stimulus_gen import generate_stimulus
from finst_video_model.scoring import classify_trial, parse_boolean_answer
from finst_video_model.vlm_client import ask_about_video
from run_pylyshyn_trial import build_question

N_OBJECTS = 2
N_CUED = 1
N_SEEDS = 20

# Shared ordinal ordering across models, so "none" (27b only) sorts below
# every effort level and the two models' points line up on one x-axis.
REASONING_LEVEL_ORDER = ["none", "minimal", "low", "medium", "high"]

REASONING_LEVELS_BY_MODEL = {
    "qwen/qwen3.8-27b": ["none", "minimal", "low", "medium", "high"],
    "qwen/qwen3.8-max": ["minimal", "low", "medium", "high"],  # reasoning is mandatory
}

RUN_NAME = "reasoning_sweep"

RESULT_FIELDS = [
    "trial_id", "model", "reasoning_level", "seed", "probe_on_target",
    "probe_is_target", "response", "predicted", "outcome",
    "reasoning_tokens", "cost", "error",
]


def reasoning_param(level: str) -> dict:
    return {"enabled": False} if level == "none" else {"effort": level}


def run_one_trial(
    batch_dir: str, model: str, reasoning_level: str,
    seed: int, probe_on_target: bool,
) -> dict:
    cfg = TrialConfig(n_objects=N_OBJECTS, n_cued=N_CUED, probe_on_target=probe_on_target, seed=seed)
    trial_dir = os.path.join(batch_dir, "trials", cfg.trial_id)
    os.makedirs(trial_dir, exist_ok=True)

    ground_truth = generate_stimulus(cfg, trial_dir)
    question = build_question(cfg)
    with open(os.path.join(trial_dir, "question.txt"), "w") as f:
        f.write(question)

    result = {
        "trial_id": cfg.trial_id, "model": model, "reasoning_level": reasoning_level,
        "seed": seed, "probe_on_target": probe_on_target,
        "probe_is_target": ground_truth["probe_is_target"],
        "video_path": ground_truth["video_path"], "question": question,
    }
    row = {
        "trial_id": cfg.trial_id, "model": model, "reasoning_level": reasoning_level,
        "seed": seed, "probe_on_target": probe_on_target,
        "probe_is_target": ground_truth["probe_is_target"],
        "response": "", "predicted": "", "outcome": "",
        "reasoning_tokens": "", "cost": "", "error": "",
    }

    try:
        response_text, usage = ask_about_video(
            ground_truth["video_path"], question, model,
            reasoning=reasoning_param(reasoning_level), return_usage=True,
        )
        predicted = parse_boolean_answer(response_text)
        outcome = (
            classify_trial(ground_truth["probe_is_target"], predicted)
            if predicted is not None else "unparseable"
        )
        result.update({"response": response_text, "predicted": predicted, "outcome": outcome, "usage": usage})
        reasoning_tokens = (usage or {}).get("completion_tokens_details", {}).get("reasoning_tokens")
        row.update({
            "response": response_text, "predicted": predicted, "outcome": outcome,
            "reasoning_tokens": reasoning_tokens, "cost": usage.get("cost") if usage else "",
        })
    except (requests.RequestException, KeyError, RuntimeError) as e:
        result["error"] = str(e)
        row["error"] = str(e)

    with open(os.path.join(trial_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)

    return row


def build_grid(models: list[str], seeds: list[int]):
    if len(seeds) % 2 != 0:
        raise ValueError(f"seeds must split evenly into matching/non-matching halves, got {len(seeds)}")
    half = len(seeds) // 2
    seeds_sorted = sorted(seeds)
    probe_on_target_by_seed = {seed: (i < half) for i, seed in enumerate(seeds_sorted)}
    grid = []
    for model in models:
        for level in REASONING_LEVELS_BY_MODEL[model]:
            for seed in seeds_sorted:
                grid.append((model, level, seed, probe_on_target_by_seed[seed]))
    return grid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=str, nargs="+", default=list(REASONING_LEVELS_BY_MODEL.keys()))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(N_SEEDS)))
    parser.add_argument("--concurrency", type=int, default=5)
    args = parser.parse_args()

    for model in args.models:
        if model not in REASONING_LEVELS_BY_MODEL:
            raise ValueError(f"No reasoning-level list for model {model!r}; add it to REASONING_LEVELS_BY_MODEL")

    batch_dir = os.path.join("data", "pylyshyn", RUN_NAME)
    results_dir = os.path.join("results", "pylyshyn", RUN_NAME)
    os.makedirs(os.path.join(batch_dir, "trials"), exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    config_path = os.path.join(results_dir, "config.json")
    if os.path.isfile(config_path):
        # Same guard as run_pylyshyn_batch.py: matching/non-matching split
        # depends on a seed's position within the whole --seeds list, so a
        # mismatched --seeds re-run would silently relabel probe_on_target.
        with open(config_path) as f:
            prior_seeds = set(json.load(f)["seeds"])
        if set(args.seeds) != prior_seeds:
            raise ValueError(
                f"{config_path} was written with seeds={sorted(prior_seeds)}, "
                f"but --seeds={sorted(set(args.seeds))} was passed -- these must match "
                f"exactly to keep matching/non-matching seed labels stable across runs."
            )
    else:
        with open(config_path, "w") as f:
            json.dump({
                "run_name": RUN_NAME, "arm": "pylyshyn", "models": args.models,
                "n_objects": N_OBJECTS, "n_cued": N_CUED,
                "reasoning_levels_by_model": REASONING_LEVELS_BY_MODEL,
                "reasoning_level_order": REASONING_LEVEL_ORDER,
                "seeds": args.seeds,
            }, f, indent=2)

    results_csv = os.path.join(results_dir, "results.csv")
    already_ran = set()
    file_exists = os.path.isfile(results_csv)
    if file_exists:
        with open(results_csv) as f:
            for r in csv.DictReader(f):
                already_ran.add((
                    r["model"], r["reasoning_level"], int(r["seed"]), r["probe_on_target"] == "True",
                ))

    grid = build_grid(args.models, args.seeds)
    pending = [spec for spec in grid if spec not in already_ran]
    total = len(grid)
    done = len(already_ran)

    with open(results_csv, "a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=RESULT_FIELDS)
        if not file_exists:
            writer.writeheader()
            csv_file.flush()

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {
                pool.submit(run_one_trial, batch_dir, model, level, seed, probe_on_target): (
                    model, level, seed, probe_on_target
                )
                for model, level, seed, probe_on_target in pending
            }
            for future in concurrent.futures.as_completed(futures):
                model, level, seed, probe_on_target = futures[future]
                row = future.result()
                done += 1
                print(
                    f"[{done}/{total}] model={model} reasoning_level={level} "
                    f"seed={seed} probe_on_target={probe_on_target} -> {row['outcome'] or row['error']}"
                )
                writer.writerow(row)
                csv_file.flush()

    print(f"\nresults.csv up to date at {results_csv}")


if __name__ == "__main__":
    main()
