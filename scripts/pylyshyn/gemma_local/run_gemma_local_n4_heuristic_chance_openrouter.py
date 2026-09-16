"""
Runs the *same* 100-trial, heuristic-neutralized pylyshyn n_objects=4 set
from Task 2 (`generate_gemma_local_n4_heuristic_chance_stimuli.py`) through
OpenRouter's hosted google/gemma-4-31b-it -- see
`claude/2026_09/2026_09_14/TODO.md`'s Task 3: "just for completion," one
host, one condition (no sweep). Reuses the exact `video.mp4` files already
on disk under `data/pylyshyn/gemma_local_n4_heuristic_chance/trials/`
(generated once in Task 2) rather than regenerating stimuli, so this is the
identical stimulus set the local npy/mp4 arms saw.

Host pinned to deepinfra-fp4 (`HOST`/`HOST_SPEC` below, same
`only`/`quantizations` filter convention as
`scripts/pylyshyn/gemma_local/run_pylyshyn_host_sweep.py` --
see [[reference-gemma-precision-mismatch]]). Originally pinned to
deepinfra-fp8 (scored 90% at n_objects=4 in the 2026-09-10 7-host sweep,
near the top), but that endpoint was persistently rate-limited on
DeepInfra's own end this session (200 OK responses with an embedded
`error` object, `"Rate limit exceeded (requests per minute)"`,
`allow_fallbacks: False` so no fallback) -- reproduced on 100/100 trials,
then again after `ask_about_video`'s new provider-error retry (see
`src/finst_video_model/vlm_client.py`) exhausted its backoff budget on a
single test call. Switched to deepinfra-fp4 (same provider, different
quantization tier -- verified healthy by a live smoke test) rather than a
different provider entirely, to stay as close as possible to the original
choice.

Same MODEL/REASONING/MAX_TOKENS as run_pylyshyn_host_sweep.py, and reuses
its `build_question` import from run_pylyshyn_trial.py so the question
wording is byte-identical to every other pylyshyn OpenRouter run.

Writes to results/pylyshyn/gemma_local/gemma_local_n4_heuristic_chance/openrouter/
{results.csv,config.json} -- results.csv written incrementally, re-running
resumes automatically (skips trial_ids already present).

Usage:
    uv run python scripts/pylyshyn/gemma_local/run_gemma_local_n4_heuristic_chance_openrouter.py
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
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_pylyshyn_trial import build_question

MODEL = "google/gemma-4-31b-it"
REASONING = {"enabled": False}
MAX_TOKENS = 8192
HOST = "deepinfra-fp4"
HOST_SPEC = {"only": ["deepinfra"], "quantizations": ["fp4"], "allow_fallbacks": False}

TRIALS_ROOT = "data/pylyshyn/gemma_local_n4_heuristic_chance/trials"
RESULTS_DIR = "results/pylyshyn/gemma_local/gemma_local_n4_heuristic_chance/openrouter"

RESULT_FIELDS = [
    "trial_id", "model", "host", "n_objects", "n_cued",
    "seed", "probe_on_target", "probe_is_target", "response", "predicted", "outcome",
    "prompt_tokens", "completion_tokens", "reasoning_tokens", "cost",
    "served_by_provider", "error",
]


def discover_trials(trials_root: str) -> list[str]:
    return sorted(
        d for d in os.listdir(trials_root)
        if os.path.isfile(os.path.join(trials_root, d, "ground_truth.json"))
    )


def run_one_trial(trials_root: str, trial_id: str) -> dict:
    trial_dir = os.path.join(trials_root, trial_id)
    with open(os.path.join(trial_dir, "ground_truth.json")) as f:
        ground_truth = json.load(f)
    cfg = TrialConfig(**ground_truth["config"])
    video_path = ground_truth["video_path"]
    question = build_question(cfg)

    row = {
        "trial_id": trial_id, "model": MODEL, "host": HOST,
        "n_objects": cfg.n_objects, "n_cued": cfg.n_cued,
        "seed": cfg.seed, "probe_on_target": cfg.probe_on_target,
        "probe_is_target": ground_truth["probe_is_target"],
        "response": "", "predicted": "", "outcome": "",
        "prompt_tokens": "", "completion_tokens": "", "reasoning_tokens": "",
        "cost": "", "served_by_provider": "", "error": "",
    }

    try:
        response_text, usage = ask_about_video(
            video_path, question, MODEL,
            reasoning=REASONING, max_tokens=MAX_TOKENS, provider=HOST_SPEC,
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
    parser.add_argument("--concurrency", type=int, default=6)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    trial_ids = discover_trials(TRIALS_ROOT)
    print(f"Discovered {len(trial_ids)} trials under {TRIALS_ROOT}")

    config_path = os.path.join(RESULTS_DIR, "config.json")
    if not os.path.isfile(config_path):
        with open(config_path, "w") as f:
            json.dump({
                "run_name": "gemma_local_n4_heuristic_chance_openrouter",
                "arm": "pylyshyn", "model": MODEL, "reasoning": REASONING,
                "host": HOST, "host_spec": HOST_SPEC, "trials_root": TRIALS_ROOT,
                "n_trials": len(trial_ids),
            }, f, indent=2)

    results_csv = os.path.join(RESULTS_DIR, "results.csv")
    file_exists = os.path.isfile(results_csv)
    already_ran = set()
    if file_exists:
        with open(results_csv) as f:
            already_ran = {r["trial_id"] for r in csv.DictReader(f)}

    pending = [tid for tid in trial_ids if tid not in already_ran]
    total = len(trial_ids)
    done = len(already_ran)
    print(f"{done}/{total} trials already done (skipping), {len(pending)} pending")

    with open(results_csv, "a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=RESULT_FIELDS)
        if not file_exists:
            writer.writeheader()
            csv_file.flush()

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {pool.submit(run_one_trial, TRIALS_ROOT, tid): tid for tid in pending}
            for future in concurrent.futures.as_completed(futures):
                trial_id = futures[future]
                row = future.result()
                done += 1
                print(
                    f"[{done}/{total}] trial={trial_id} "
                    f"-> {row['outcome'] or row['error']} (served_by={row['served_by_provider']})"
                )
                writer.writerow(row)
                csv_file.flush()

    print(f"\nresults.csv up to date at {results_csv}")

    with open(results_csv) as f:
        all_rows = list(csv.DictReader(f))

    mismatches = [
        r for r in all_rows
        if r["served_by_provider"] and "deepinfra" not in r["served_by_provider"].lower()
    ]
    if mismatches:
        print(f"\n*** WARNING: {len(mismatches)} rows' served_by_provider doesn't match host={HOST} ***")
        for r in mismatches[:10]:
            print(f"  trial_id={r['trial_id']} served_by_provider={r['served_by_provider']}")

    scorable = [
        {"probe_is_target": r["probe_is_target"] == "True", "predicted": (
            None if r["predicted"] == "" else r["predicted"] == "True"
        )}
        for r in all_rows
    ]
    stats = compute_d_prime(scorable)
    print(
        f"\nhost={HOST} n_objects=4: d'={stats['d_prime']:.2f} "
        f"(n={len(scorable)}, unparseable={stats['n_unparseable']})"
    )


if __name__ == "__main__":
    main()
