"""
Runs the 100-trial, heuristic-always-wrong pylyshyn n_objects=3 set from
`generate_hardest_stimuli.py` through 5 OpenRouter models at
reasoning="medium" -- `claude/2026_09/2026_09_16/TODO.md`'s Part 4
reasoning-on pass. Sibling to `run_hardest_stimuli_sweep.py` (reasoning
off, 4 models); this one adds `google/gemini-3.8-flash` back in, since its
reasoning is mandatory on OpenRouter (can't run it with reasoning off at
all -- see that script's docstring) but works fine at effort="medium".

Model set here (5): gemma-4-31b-it, qwen3.6-plus, kimi-k3, qwen3.8-27b,
gemini-3.8-flash. gemma pinned to `modelrun` only for this condition
specifically (per [[project-status-2026-09-16]]: DeepInfra took 298-495s/
call under reasoning="medium" vs. modelrun's 4.5s, a real host-latency
issue, not a cost issue -- same fix as Part 2's thinking sweep).

Writes to results/pylyshyn/hard_stimuli/hardest_stimuli_thinking_sweep/
{results.csv,config.json} -- resumable.

Usage:
    uv run python scripts/pylyshyn/hard_stimuli/run_hardest_stimuli_thinking_sweep.py
"""

import argparse
import concurrent.futures
import csv
import json
import os

import requests

from finst_video_model.pylyshyn.config import TrialConfig
from finst_video_model.scoring import classify_trial, compute_d_prime, parse_boolean_answer
from finst_video_model.vlm_client import ask_about_video

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_pylyshyn_trial import build_question

REASONING_EFFORT = "medium"
MODEL_REASONING = {
    "google/gemma-4-31b-it": {"effort": REASONING_EFFORT},
    "qwen/qwen3.6-plus": {"effort": REASONING_EFFORT},
    "moonshotai/kimi-k3": {"effort": REASONING_EFFORT},
    "qwen/qwen3.8-27b": {"effort": REASONING_EFFORT},
    "google/gemini-3.8-flash": {"effort": REASONING_EFFORT},
}
MODEL_PROVIDER = {
    "moonshotai/kimi-k3": {"only": ["moonshotai"], "allow_fallbacks": False},
    "google/gemma-4-31b-it": {"only": ["modelrun"], "allow_fallbacks": False},
}
MAX_TOKENS = 16384

TRIALS_ROOT = "data/pylyshyn/hardest_stimuli_sweep/trials"
RESULTS_DIR = "results/pylyshyn/hard_stimuli/hardest_stimuli_thinking_sweep"

RESULT_FIELDS = [
    "trial_id", "model", "n_objects", "n_cued",
    "seed", "probe_on_target", "probe_is_target", "response", "predicted", "outcome",
    "prompt_tokens", "completion_tokens", "reasoning_tokens", "cost",
    "served_by_provider", "error",
]


def discover_trials(trials_root: str) -> list[str]:
    return sorted(
        d for d in os.listdir(trials_root)
        if os.path.isfile(os.path.join(trials_root, d, "ground_truth.json"))
    )


def run_one_trial(trials_root: str, model: str, trial_id: str) -> dict:
    trial_dir = os.path.join(trials_root, trial_id)
    with open(os.path.join(trial_dir, "ground_truth.json")) as f:
        ground_truth = json.load(f)
    cfg = TrialConfig(**ground_truth["config"])
    video_path = os.path.join(trial_dir, "video_stretched.mp4")
    question = build_question(cfg)

    row = {
        "trial_id": trial_id, "model": model,
        "n_objects": cfg.n_objects, "n_cued": cfg.n_cued,
        "seed": cfg.seed, "probe_on_target": cfg.probe_on_target,
        "probe_is_target": ground_truth["probe_is_target"],
        "response": "", "predicted": "", "outcome": "",
        "prompt_tokens": "", "completion_tokens": "", "reasoning_tokens": "",
        "cost": "", "served_by_provider": "", "error": "",
    }

    try:
        response_text, usage = ask_about_video(
            video_path, question, model,
            reasoning=MODEL_REASONING[model], max_tokens=MAX_TOKENS,
            provider=MODEL_PROVIDER.get(model),
            return_usage=True,
        )
        predicted = parse_boolean_answer(response_text)
        outcome = (
            classify_trial(ground_truth["probe_is_target"], predicted)
            if predicted is not None else "unparseable"
        )
        row.update({
            "response": response_text, "predicted": predicted, "outcome": outcome,
            "prompt_tokens": (usage or {}).get("prompt_tokens"),
            "completion_tokens": (usage or {}).get("completion_tokens"),
            "reasoning_tokens": (usage or {}).get("completion_tokens_details", {}).get("reasoning_tokens"),
            "cost": (usage or {}).get("cost"),
            "served_by_provider": (usage or {}).get("served_by_provider"),
        })
    except (requests.RequestException, KeyError, RuntimeError) as e:
        row["error"] = str(e)

    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=str, nargs="+", default=list(MODEL_REASONING.keys()))
    parser.add_argument("--concurrency", type=int, default=16)
    args = parser.parse_args()

    for model in args.models:
        if model not in MODEL_REASONING:
            raise ValueError(f"No reasoning param for model {model!r}; add it to MODEL_REASONING")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    trial_ids = discover_trials(TRIALS_ROOT)
    print(f"Discovered {len(trial_ids)} trials under {TRIALS_ROOT}")

    config_path = os.path.join(RESULTS_DIR, "config.json")
    if not os.path.isfile(config_path):
        with open(config_path, "w") as f:
            json.dump({
                "run_name": "hardest_stimuli_thinking_sweep", "arm": "pylyshyn",
                "models": args.models, "model_reasoning": MODEL_REASONING,
                "model_provider": MODEL_PROVIDER, "variant": "stretched",
                "max_tokens": MAX_TOKENS,
                "trials_root": TRIALS_ROOT, "n_trials": len(trial_ids),
            }, f, indent=2)

    results_csv = os.path.join(RESULTS_DIR, "results.csv")
    file_exists = os.path.isfile(results_csv)
    already_ran = set()
    if file_exists:
        with open(results_csv) as f:
            already_ran = {(r["model"], r["trial_id"]) for r in csv.DictReader(f)}

    grid = [(model, tid) for model in args.models for tid in trial_ids]
    pending = [(model, tid) for model, tid in grid if (model, tid) not in already_ran]
    total = len(grid)
    done = len(already_ran)
    print(f"{done}/{total} (model, trial) pairs already done (skipping), {len(pending)} pending")

    with open(results_csv, "a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=RESULT_FIELDS)
        if not file_exists:
            writer.writeheader()
            csv_file.flush()

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {
                pool.submit(run_one_trial, TRIALS_ROOT, model, tid): (model, tid)
                for model, tid in pending
            }
            for future in concurrent.futures.as_completed(futures):
                model, trial_id = futures[future]
                row = future.result()
                done += 1
                print(
                    f"[{done}/{total}] model={model} trial={trial_id} "
                    f"-> {row['outcome'] or row['error']} (reasoning_tokens={row['reasoning_tokens']}, "
                    f"served_by={row['served_by_provider']})"
                )
                writer.writerow(row)
                csv_file.flush()

    print(f"\nresults.csv up to date at {results_csv}")

    with open(results_csv) as f:
        all_rows = list(csv.DictReader(f))
    for model in args.models:
        rows = [
            {"probe_is_target": r["probe_is_target"] == "True", "predicted": (
                None if r["predicted"] == "" else r["predicted"] == "True"
            )}
            for r in all_rows if r["model"] == model
        ]
        if not rows:
            continue
        stats = compute_d_prime(rows)
        total_cost = sum(float(r["cost"]) for r in all_rows if r["model"] == model and r["cost"])
        print(
            f"model={model}: d'={stats['d_prime']:.2f} (n={len(rows)}, "
            f"unparseable={stats['n_unparseable']}), total_cost=${total_cost:.4f}"
        )


if __name__ == "__main__":
    main()
