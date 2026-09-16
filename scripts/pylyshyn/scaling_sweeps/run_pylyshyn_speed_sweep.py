"""
Batch runner for claude/2026_08/2026_08_27/TODO.md's "Task 2": does
pylyshyn's motion speed/redirect cadence affect qwen3.6-plus, the strongest
no-reasoning model from yesterday's Part 4
(claude/2026_08/2026_08_26/TODO.md, see [[project-status-2026-08-26]])?
Model and reasoning are fixed; the swept axes are redirect_condition x
speed_condition x video variant (native/stretched), reusing
run_pylyshyn_stretch_sweep.py's stimulus-gen/scoring/resume machinery
almost verbatim.

REDIRECT_CONDITIONS/SPEED_CONDITIONS: "slow" is the 2026-08-26 "slow it
down" pass baked into TrialConfig's current defaults
(redirect_min_s=redirect_max_s=2.0, speed_min/max_px_s=16.0/33.0); "fast"
reverts redirect to the original 1.0s cadence and doubles the speed bounds
back up, per the user's explicit request 2026-08-27.

STRETCH_ENCODE_FPS is unaffected by either swept axis (both only change
motion physics, not total_frames/fps), so it's the same constant
run_pylyshyn_stretch_sweep.py derives.

The "slow, slow" corner here is identical in content-generating config to
what qwen3.6-plus/none already ran in
results/pylyshyn/reasoning_sweeps/stretch_sweep/results.csv (32 rows, both variants) --
this sweep still re-runs it fresh rather than splicing those rows in, to
keep the two results.csv schemas (this one has redirect_condition/
speed_condition columns, not reasoning_level) independent and simple.

Every trial's artifacts land in data/pylyshyn/speed_sweep/trials/<trial_id>/
(gitignored); results.csv/config.json (tracked) go to
results/pylyshyn/scaling_sweeps/speed_sweep/. results.csv is written incrementally and
re-running resumes automatically (already-run (redirect_condition,
speed_condition, variant, seed, probe_on_target) combinations are skipped).

Usage:
    uv run python scripts/pylyshyn/scaling_sweeps/run_pylyshyn_speed_sweep.py
"""

import argparse
import concurrent.futures
import csv
import json
import os

import requests

from finst_video_model.debug_circular.stimulus_gen import write_mp4
from finst_video_model.pylyshyn.config import TrialConfig
from finst_video_model.pylyshyn.stimulus_gen import generate_stimulus
from finst_video_model.scoring import classify_trial, compute_d_prime, parse_boolean_answer
from finst_video_model.vlm_client import ask_about_video
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_pylyshyn_trial import build_question

MODEL = "qwen/qwen3.6-plus"
N_OBJECTS = 3
N_CUED = 1
N_SEEDS = 16
MAX_TOKENS = 8192

# "slow" == TrialConfig()'s current 2026-08-26 defaults; "fast" reverts
# redirect to the original 1.0s cadence and doubles the halved speed bounds
# back up (see module docstring).
REDIRECT_CONDITIONS = {"slow": 2.0, "fast": 1.0}
SPEED_CONDITIONS = {"slow": (16.0, 33.0), "fast": (32.0, 66.0)}
VARIANTS = ["native", "stretched"]

RUN_NAME = "speed_sweep"

STRETCH_TARGET_DURATION_S = 50.0
_NATIVE_TOTAL_FRAMES = TrialConfig(n_objects=N_OBJECTS, n_cued=N_CUED).total_frames
STRETCH_ENCODE_FPS = _NATIVE_TOTAL_FRAMES / STRETCH_TARGET_DURATION_S

RESULT_FIELDS = [
    "trial_id", "model", "redirect_condition", "speed_condition", "variant",
    "n_objects", "n_cued", "seed", "probe_on_target", "probe_is_target",
    "response", "predicted", "outcome",
    "prompt_tokens", "completion_tokens", "reasoning_tokens", "cost", "error",
]


def run_one_trial(
    batch_dir: str, redirect_condition: str, speed_condition: str, variant: str,
    seed: int, probe_on_target: bool,
) -> dict:
    redirect_s = REDIRECT_CONDITIONS[redirect_condition]
    speed_min, speed_max = SPEED_CONDITIONS[speed_condition]
    cfg = TrialConfig(
        n_objects=N_OBJECTS, n_cued=N_CUED, probe_on_target=probe_on_target, seed=seed,
        redirect_min_s=redirect_s, redirect_max_s=redirect_s,
        speed_min_px_s=speed_min, speed_max_px_s=speed_max,
    )
    trial_dir = os.path.join(batch_dir, "trials", cfg.trial_id)
    os.makedirs(trial_dir, exist_ok=True)

    ground_truth, frames = generate_stimulus(cfg, trial_dir)
    if variant == "stretched":
        video_path = os.path.join(trial_dir, "video_stretched.mp4")
        write_mp4(frames, video_path, fps=STRETCH_ENCODE_FPS)
    else:
        video_path = ground_truth["video_path"]

    question = build_question(cfg)
    with open(os.path.join(trial_dir, "question.txt"), "w") as f:
        f.write(question)

    result = {
        "trial_id": cfg.trial_id, "model": MODEL,
        "redirect_condition": redirect_condition, "speed_condition": speed_condition,
        "variant": variant, "n_objects": N_OBJECTS, "n_cued": N_CUED,
        "seed": seed, "probe_on_target": probe_on_target,
        "probe_is_target": ground_truth["probe_is_target"],
        "video_path": video_path, "question": question,
    }
    row = {
        "trial_id": cfg.trial_id, "model": MODEL,
        "redirect_condition": redirect_condition, "speed_condition": speed_condition,
        "variant": variant, "n_objects": N_OBJECTS, "n_cued": N_CUED,
        "seed": seed, "probe_on_target": probe_on_target,
        "probe_is_target": ground_truth["probe_is_target"],
        "response": "", "predicted": "", "outcome": "",
        "prompt_tokens": "", "completion_tokens": "", "reasoning_tokens": "",
        "cost": "", "error": "",
    }

    try:
        response_text, usage = ask_about_video(
            video_path, question, MODEL,
            reasoning={"enabled": False}, max_tokens=MAX_TOKENS,
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
        })
    except (requests.RequestException, KeyError, RuntimeError) as e:
        result["error"] = str(e)
        row["error"] = str(e)

    with open(os.path.join(trial_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)

    return row


def build_grid(seeds: list[int]):
    if len(seeds) % 2 != 0:
        raise ValueError(f"seeds must split evenly into matching/non-matching halves, got {len(seeds)}")
    half = len(seeds) // 2
    seeds_sorted = sorted(seeds)
    probe_on_target_by_seed = {seed: (i < half) for i, seed in enumerate(seeds_sorted)}
    grid = []
    for redirect_condition in REDIRECT_CONDITIONS:
        for speed_condition in SPEED_CONDITIONS:
            for variant in VARIANTS:
                for seed in seeds_sorted:
                    grid.append((redirect_condition, speed_condition, variant, seed, probe_on_target_by_seed[seed]))
    return grid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(N_SEEDS)))
    parser.add_argument("--concurrency", type=int, default=5)
    args = parser.parse_args()

    print(
        f"Native clip: {_NATIVE_TOTAL_FRAMES} frames @ {TrialConfig().fps}fps "
        f"(~{_NATIVE_TOTAL_FRAMES / TrialConfig().fps:.1f}s). "
        f"Stretched encode: {STRETCH_ENCODE_FPS:.2f}fps (~{STRETCH_TARGET_DURATION_S:.0f}s nominal)."
    )

    batch_dir = os.path.join("data", "pylyshyn", RUN_NAME)
    results_dir = os.path.join("results", "pylyshyn", "scaling_sweeps", RUN_NAME)
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
                "n_objects": N_OBJECTS, "n_cued": N_CUED,
                "redirect_conditions": REDIRECT_CONDITIONS, "speed_conditions": SPEED_CONDITIONS,
                "variants": VARIANTS, "stretch_encode_fps": STRETCH_ENCODE_FPS,
                "seeds": args.seeds,
            }, f, indent=2)

    results_csv = os.path.join(results_dir, "results.csv")
    already_ran = set()
    file_exists = os.path.isfile(results_csv)
    if file_exists:
        with open(results_csv) as f:
            for r in csv.DictReader(f):
                already_ran.add((
                    r["redirect_condition"], r["speed_condition"], r["variant"],
                    int(r["seed"]), r["probe_on_target"] == "True",
                ))

    grid = build_grid(args.seeds)
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
                pool.submit(run_one_trial, batch_dir, redirect_condition, speed_condition, variant, seed, probe_on_target): (
                    redirect_condition, speed_condition, variant, seed, probe_on_target
                )
                for redirect_condition, speed_condition, variant, seed, probe_on_target in pending
            }
            for future in concurrent.futures.as_completed(futures):
                redirect_condition, speed_condition, variant, seed, probe_on_target = futures[future]
                row = future.result()
                done += 1
                print(
                    f"[{done}/{total}] redirect={redirect_condition} speed={speed_condition} variant={variant} "
                    f"seed={seed} probe_on_target={probe_on_target} -> {row['outcome'] or row['error']}"
                )
                writer.writerow(row)
                csv_file.flush()

    print(f"\nresults.csv up to date at {results_csv}")

    with open(results_csv) as f:
        all_rows = list(csv.DictReader(f))
    for redirect_condition in REDIRECT_CONDITIONS:
        for speed_condition in SPEED_CONDITIONS:
            for variant in VARIANTS:
                rows = [
                    {"probe_is_target": r["probe_is_target"] == "True", "predicted": (
                        None if r["predicted"] == "" else r["predicted"] == "True"
                    )}
                    for r in all_rows
                    if r["redirect_condition"] == redirect_condition
                    and r["speed_condition"] == speed_condition and r["variant"] == variant
                ]
                if not rows:
                    continue
                stats = compute_d_prime(rows)
                print(
                    f"redirect={redirect_condition} speed={speed_condition} variant={variant}: "
                    f"d'={stats['d_prime']:.2f} (n={len(rows)}, unparseable={stats['n_unparseable']})"
                )


if __name__ == "__main__":
    main()
