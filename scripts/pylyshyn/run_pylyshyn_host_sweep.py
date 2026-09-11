"""
Batch runner for `claude/2026_09/2026_09_10/TODO.md`'s "Testing different
hosts of gemma4" task: does `google/gemma-4-31b-it`'s accuracy on pylyshyn
(n_objects sweep, fast redirect/fast speed, native only) vary across
different OpenRouter *providers* serving the identical model slug? Sibling
script to `scripts/debug_circular/run_debug_circular_host_sweep.py` -- see
its module docstring for the full [[reference-gemma-precision-mismatch]]
motivation and the `deepinfra-fp4`/`deepinfra-fp8` `quantizations`-filter
disambiguation, both identical here.

`host` is a new grid axis, kept separate from `model` (fixed at
`google/gemma-4-31b-it` throughout). Every row logs `served_by_provider`
to catch a silent provider fallback. Mirrors
`run_pylyshyn_speed_sweep_models.py`'s structure (fast redirect_s=1.0 /
fast speed_px_s=32-66, matching [[project-status-2026-09-10]]'s local-vs-
OpenRouter comparison condition), but with a host axis instead of a model
axis, and native-only (no `variant` loop).

Every trial's artifacts land in data/pylyshyn/host_sweep/trials/
(gitignored); results.csv/config.json (tracked) go to
results/pylyshyn/host_sweep/. results.csv is written incrementally and
re-running resumes automatically.

Usage:
    uv run python scripts/pylyshyn/run_pylyshyn_host_sweep.py
    uv run python scripts/pylyshyn/run_pylyshyn_host_sweep.py --hosts coreweave crusoe
"""

import argparse
import concurrent.futures
import csv
import json
import os

import requests

from finst_video_model.pylyshyn.config import TrialConfig
from finst_video_model.pylyshyn.stimulus_gen import generate_stimulus
from finst_video_model.scoring import classify_trial, compute_d_prime, parse_boolean_answer
from finst_video_model.vlm_client import ask_about_video
from run_pylyshyn_trial import build_question

MODEL = "google/gemma-4-31b-it"
REASONING = {"enabled": False}
N_CUED = 1
N_SEEDS = 20
MAX_TOKENS = 8192
N_OBJECTS_VALUES = [3, 4, 5]
REDIRECT_S = 1.0
SPEED_MIN_PX_S = 32.0
SPEED_MAX_PX_S = 66.0
RUN_NAME = "host_sweep"

# See scripts/debug_circular/run_debug_circular_host_sweep.py's module
# docstring for why deepinfra-fp4/deepinfra-fp8 use the `quantizations`
# filter.
HOSTS = {
    "deepinfra-fp4": {"only": ["deepinfra"], "quantizations": ["fp4"], "allow_fallbacks": False},
    "deepinfra-fp8": {"only": ["deepinfra"], "quantizations": ["fp8"], "allow_fallbacks": False},
    "coreweave": {"only": ["coreweave"], "allow_fallbacks": False},
    "crusoe": {"only": ["crusoe"], "allow_fallbacks": False},
    "parasail": {"only": ["parasail"], "allow_fallbacks": False},
    "together": {"only": ["together"], "allow_fallbacks": False},
    "modelrun": {"only": ["modelrun"], "allow_fallbacks": False},
}
# Snapshot of OpenRouter's live /models/google/gemma-4-31b-it/endpoints
# response, checked 2026-09-10 -- see reference_gemma_precision_mismatch memory.
HOST_QUANTIZATION = {
    "deepinfra-fp4": "fp4", "deepinfra-fp8": "fp8", "coreweave": "fp4",
    "crusoe": "unknown", "parasail": "fp8", "together": "unknown", "modelrun": "fp4",
}

RESULT_FIELDS = [
    "trial_id", "model", "host", "n_objects", "n_cued",
    "seed", "probe_on_target", "probe_is_target", "response", "predicted", "outcome",
    "prompt_tokens", "completion_tokens", "reasoning_tokens", "cost",
    "served_by_provider", "error",
]


def run_one_trial(batch_dir: str, host: str, n_objects: int, seed: int, probe_on_target: bool) -> dict:
    cfg = TrialConfig(
        n_objects=n_objects, n_cued=N_CUED, probe_on_target=probe_on_target, seed=seed,
        redirect_min_s=REDIRECT_S, redirect_max_s=REDIRECT_S,
        speed_min_px_s=SPEED_MIN_PX_S, speed_max_px_s=SPEED_MAX_PX_S,
    )
    trial_dir = os.path.join(batch_dir, "trials", f"{host}_{cfg.trial_id}")
    os.makedirs(trial_dir, exist_ok=True)

    ground_truth, _frames = generate_stimulus(cfg, trial_dir)
    video_path = ground_truth["video_path"]

    question = build_question(cfg)
    with open(os.path.join(trial_dir, "question.txt"), "w") as f:
        f.write(question)

    row = {
        "trial_id": cfg.trial_id, "model": MODEL, "host": host,
        "n_objects": n_objects, "n_cued": N_CUED, "seed": seed, "probe_on_target": probe_on_target,
        "probe_is_target": ground_truth["probe_is_target"],
        "response": "", "predicted": "", "outcome": "",
        "prompt_tokens": "", "completion_tokens": "", "reasoning_tokens": "",
        "cost": "", "served_by_provider": "", "error": "",
    }
    result = dict(row, video_path=video_path, question=question)

    try:
        response_text, usage = ask_about_video(
            video_path, question, MODEL,
            reasoning=REASONING, max_tokens=MAX_TOKENS, provider=HOSTS[host],
            return_usage=True,
        )
        predicted = parse_boolean_answer(response_text)
        outcome = (
            classify_trial(ground_truth["probe_is_target"], predicted)
            if predicted is not None else "unparseable"
        )
        result.update({"response": response_text, "predicted": predicted, "outcome": outcome, "usage": usage})
        row.update({
            "response": response_text, "predicted": predicted, "outcome": outcome,
            "prompt_tokens": (usage or {}).get("prompt_tokens"),
            "completion_tokens": (usage or {}).get("completion_tokens"),
            "reasoning_tokens": (usage or {}).get("completion_tokens_details", {}).get("reasoning_tokens"),
            "cost": (usage or {}).get("cost"),
            "served_by_provider": (usage or {}).get("served_by_provider"),
        })
    except (requests.RequestException, KeyError, RuntimeError) as e:
        result["error"] = str(e)
        row["error"] = str(e)

    with open(os.path.join(trial_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)

    return row


def build_grid(hosts: list[str], n_objects_values: list[int], seeds: list[int]):
    if len(seeds) % 2 != 0:
        raise ValueError(f"seeds must split evenly into matching/non-matching halves, got {len(seeds)}")
    half = len(seeds) // 2
    seeds_sorted = sorted(seeds)
    probe_on_target_by_seed = {seed: (i < half) for i, seed in enumerate(seeds_sorted)}
    grid = []
    for host in hosts:
        for n_objects in n_objects_values:
            for seed in seeds_sorted:
                grid.append((host, n_objects, seed, probe_on_target_by_seed[seed]))
    return grid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hosts", type=str, nargs="+", default=list(HOSTS.keys()), choices=list(HOSTS.keys()))
    parser.add_argument("--n-objects", type=int, nargs="+", default=N_OBJECTS_VALUES)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(N_SEEDS)))
    parser.add_argument("--concurrency", type=int, default=6)
    args = parser.parse_args()

    batch_dir = os.path.join("data", "pylyshyn", RUN_NAME)
    results_dir = os.path.join("results", "pylyshyn", RUN_NAME)
    os.makedirs(os.path.join(batch_dir, "trials"), exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    config_path = os.path.join(results_dir, "config.json")
    if os.path.isfile(config_path):
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
                "run_name": RUN_NAME, "arm": "pylyshyn", "model": MODEL,
                "reasoning": REASONING, "n_objects": args.n_objects, "n_cued": N_CUED,
                "redirect_s": REDIRECT_S, "speed_px_s": [SPEED_MIN_PX_S, SPEED_MAX_PX_S],
                "variant": "native", "hosts": HOSTS, "host_quantization": HOST_QUANTIZATION,
                "seeds": args.seeds,
            }, f, indent=2)

    results_csv = os.path.join(results_dir, "results.csv")
    already_ran = set()
    file_exists = os.path.isfile(results_csv)
    if file_exists:
        with open(results_csv) as f:
            for r in csv.DictReader(f):
                already_ran.add((
                    r["host"], int(r["n_objects"]), int(r["seed"]), r["probe_on_target"] == "True",
                ))

    grid = build_grid(args.hosts, args.n_objects, args.seeds)
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
                pool.submit(run_one_trial, batch_dir, host, n_objects, seed, probe_on_target): (
                    host, n_objects, seed, probe_on_target
                )
                for host, n_objects, seed, probe_on_target in pending
            }
            for future in concurrent.futures.as_completed(futures):
                host, n_objects, seed, probe_on_target = futures[future]
                row = future.result()
                done += 1
                print(
                    f"[{done}/{total}] host={host} n_objects={n_objects} "
                    f"seed={seed} probe_on_target={probe_on_target} "
                    f"-> {row['outcome'] or row['error']} (served_by={row['served_by_provider']})"
                )
                writer.writerow(row)
                csv_file.flush()

    print(f"\nresults.csv up to date at {results_csv}")

    with open(results_csv) as f:
        all_rows = list(csv.DictReader(f))

    mismatches = [
        r for r in all_rows
        if r["served_by_provider"] and r["served_by_provider"].lower() not in r["host"].lower()
        and r["host"].split("-")[0] not in r["served_by_provider"].lower()
    ]
    if mismatches:
        print(f"\n*** WARNING: {len(mismatches)} rows' served_by_provider doesn't match their intended host ***")
        for r in mismatches[:10]:
            print(f"  trial_id={r['trial_id']} host={r['host']} served_by_provider={r['served_by_provider']}")

    for host in args.hosts:
        for n_objects in args.n_objects:
            rows = [
                {"probe_is_target": r["probe_is_target"] == "True", "predicted": (
                    None if r["predicted"] == "" else r["predicted"] == "True"
                )}
                for r in all_rows
                if r["host"] == host and int(r["n_objects"]) == n_objects
            ]
            if not rows:
                continue
            stats = compute_d_prime(rows)
            print(
                f"host={host} n_objects={n_objects}: "
                f"d'={stats['d_prime']:.2f} (n={len(rows)}, unparseable={stats['n_unparseable']})"
            )


if __name__ == "__main__":
    main()
