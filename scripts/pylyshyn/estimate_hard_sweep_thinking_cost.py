"""
Approximates the dollar cost of `run_hard_stimuli_thinking_sweep.py`'s full
100-trial x 5-model, reasoning="medium" sweep before running it --
`claude/2026_09/2026_09_16/TODO.md`'s Part 2, same "approximate the cost
first" convention established in Part 1's
`estimate_hard_sweep_cost.py` (live pricing sanity check + real pilot
trials with `return_usage=True` for actual measured $/trial, since
reasoning-token spend is not reliably predictable from `max_tokens` or
model size alone -- some models scale reasoning tokens organically with
effort, others pin to small fixed budgets, per
[[reference-openrouter-reasoning-max-tokens]]).

Expected to cost noticeably more per trial than Part 1's no-reasoning
sweep, since reasoning tokens are billed at each model's completion rate
(the same rate as the visible answer) on top of whatever the model
generates for the answer itself -- this is exactly why the user asked for
an estimate before committing to the real run.

Pilot trials reuse PILOT_N_TRIALS of the same 100-trial stimulus set and
get appended to the real sweep's results.csv (same resumability convention
as Part 1's estimator), so no pilot spend is wasted once the full sweep
runs.

Usage:
    uv run python scripts/pylyshyn/estimate_hard_sweep_thinking_cost.py
    uv run python scripts/pylyshyn/estimate_hard_sweep_thinking_cost.py --pilot-n 5
"""

import argparse
import csv
import json
import os

import requests
from dotenv import load_dotenv

from finst_video_model.pylyshyn.config import TrialConfig
from finst_video_model.scoring import classify_trial, parse_boolean_answer
from finst_video_model.vlm_client import ask_about_video
from run_hard_stimuli_thinking_sweep import (
    MODEL_PROVIDER, MODEL_REASONING, MAX_TOKENS, RESULT_FIELDS, RESULTS_DIR, TRIALS_ROOT, discover_trials,
)
from run_pylyshyn_trial import build_question

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
N_TRIALS_PLANNED = 100
OUT_PATH = os.path.join(RESULTS_DIR, "cost_estimate.json")
RESULTS_CSV = os.path.join(RESULTS_DIR, "results.csv")


def fetch_live_pricing(model: str) -> dict | None:
    author, slug = model.split("/", 1)
    try:
        response = requests.get(f"{OPENROUTER_BASE_URL}/models/{author}/{slug}/endpoints", timeout=30)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        print(f"  (live pricing lookup failed for {model}: {e})")
        return None

    endpoints = data.get("data", {}).get("endpoints", [])
    if not endpoints:
        return None
    completion_prices = [float(ep["pricing"]["completion"]) for ep in endpoints if "pricing" in ep]
    prompt_prices = [float(ep["pricing"]["prompt"]) for ep in endpoints if "pricing" in ep]
    if not prompt_prices:
        return None
    return {
        "n_endpoints": len(endpoints),
        "prompt_usd_per_token_min": min(prompt_prices),
        "prompt_usd_per_token_max": max(prompt_prices),
        "completion_usd_per_token_min": min(completion_prices),
        "completion_usd_per_token_max": max(completion_prices),
    }


def _load_already_ran() -> set:
    if not os.path.isfile(RESULTS_CSV):
        return set()
    with open(RESULTS_CSV) as f:
        return {(r["model"], r["trial_id"]) for r in csv.DictReader(f)}


def _append_row(row: dict):
    file_exists = os.path.isfile(RESULTS_CSV)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def run_pilot(model: str, trial_ids: list[str], already_ran: set) -> dict:
    costs, prompt_tokens, completion_tokens, reasoning_tokens = [], [], [], []
    for trial_id in trial_ids:
        if (model, trial_id) in already_ran:
            print(f"    trial={trial_id} already in results.csv, skipping pilot call")
            continue

        trial_dir = os.path.join(TRIALS_ROOT, trial_id)
        with open(os.path.join(trial_dir, "ground_truth.json")) as f:
            ground_truth = json.load(f)
        cfg = TrialConfig(**ground_truth["config"])
        video_path = os.path.join(trial_dir, "video_stretched.mp4")
        question = build_question(cfg)

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
        cost = (usage or {}).get("cost")
        if cost is not None:
            costs.append(float(cost))
        prompt_tokens.append((usage or {}).get("prompt_tokens"))
        completion_tokens.append((usage or {}).get("completion_tokens"))
        r_tokens = (usage or {}).get("completion_tokens_details", {}).get("reasoning_tokens")
        reasoning_tokens.append(r_tokens)
        print(
            f"    trial={trial_id} cost=${cost} prompt_tokens={prompt_tokens[-1]} "
            f"completion_tokens={completion_tokens[-1]} reasoning_tokens={r_tokens}"
        )

        _append_row({
            "trial_id": trial_id, "model": model,
            "n_objects": cfg.n_objects, "n_cued": cfg.n_cued,
            "seed": cfg.seed, "probe_on_target": cfg.probe_on_target,
            "probe_is_target": ground_truth["probe_is_target"],
            "response": response_text, "predicted": predicted, "outcome": outcome,
            "prompt_tokens": (usage or {}).get("prompt_tokens"),
            "completion_tokens": (usage or {}).get("completion_tokens"),
            "reasoning_tokens": r_tokens,
            "cost": cost, "served_by_provider": (usage or {}).get("served_by_provider"), "error": "",
        })
        already_ran.add((model, trial_id))

    mean_cost = sum(costs) / len(costs) if costs else None
    return {
        "n_pilot_trials": len(trial_ids),
        "n_with_cost": len(costs),
        "pilot_costs_usd": costs,
        "mean_cost_per_trial_usd": mean_cost,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "estimated_total_usd": (mean_cost * N_TRIALS_PLANNED) if mean_cost is not None else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-n", type=int, default=3)
    parser.add_argument("--models", type=str, nargs="+", default=list(MODEL_REASONING.keys()))
    args = parser.parse_args()

    trial_ids = discover_trials(TRIALS_ROOT)[: args.pilot_n]
    if len(trial_ids) < args.pilot_n:
        raise RuntimeError(f"only found {len(trial_ids)} trials under {TRIALS_ROOT}")
    print(f"Pilot trial_ids ({args.pilot_n}): {trial_ids}\n")

    already_ran = _load_already_ran()

    report = {}
    total_estimated = 0.0
    for model in args.models:
        print(f"=== {model} (reasoning={MODEL_REASONING[model]}) ===")
        pricing = fetch_live_pricing(model)
        if pricing:
            print(f"  live pricing: {pricing}")
        pilot = run_pilot(model, trial_ids, already_ran)
        if pilot["mean_cost_per_trial_usd"] is not None:
            print(
                f"  mean measured cost/trial=${pilot['mean_cost_per_trial_usd']:.5f} -> "
                f"estimated ${pilot['estimated_total_usd']:.2f} for {N_TRIALS_PLANNED} trials"
            )
            total_estimated += pilot["estimated_total_usd"]
        else:
            print("  no cost data returned by OpenRouter for this model's pilot calls")
        report[model] = {"live_pricing": pricing, "pilot": pilot}
        print()

    report["_summary"] = {
        "n_trials_planned_per_model": N_TRIALS_PLANNED,
        "n_models": len(args.models),
        "total_estimated_usd": total_estimated,
    }
    print(f"Total estimated cost across {len(args.models)} models x {N_TRIALS_PLANNED} trials: ${total_estimated:.2f}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
