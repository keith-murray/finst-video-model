"""
Batch runner for claude/2026_08/2026_08_27/TODO.md's "Task 4": follow-up on
Task 3's surprising finding that `google/gemma-4-31b-it` (a 31B dense
model) ties qwen3.6-plus and beats much larger models on pylyshyn (see
[[project-status-2026-08-27]]). Tests the same model on the debug_circular
stimulus (rigid-ring rotation, a different -- interpolable, per
[[feedback-scope-model-comparisons-to-stimulus]] -- motion type than
pylyshyn's random walk) across a rotation_deg sweep, native vs. stretched.

Confirmed live 2026-08-27 (see [[reference-gemma-fixed-frame-budget]])
that gemma-4-31b-it's OpenRouter provider extracts a fixed-size frame
sample regardless of real video length or declared duration/fps -- so the
native/stretched duration-stretch trick was expected to do nothing here,
and the first run (n_objects=3, ROTATION_DEGS default below) confirmed
that. That run also surfaced an unpredicted V-shaped accuracy curve --
near-ceiling at the extremes (40/360 deg) but *below chance* in the middle
(200-240 deg) -- consistent with a position-heuristic failure mode (see
[[project-status-2026-07-21]]'s older, unrelated-stimulus version of the
same idea): with n_objects=3 equally spaced at 120 deg apart, a rotation
near a multiple of 120 deg moves the cued cross into another cross's
original slot, which would fool a model tracking position rather than
identity.

`--n-objects`/`--rotation-degs`/`--variants`/`--run-name` were added after
that first run specifically to test this hypothesis on `n_objects=4`
(120 deg spacing -> 90 deg spacing) without touching the original run's
data -- pass a different `--run-name` for the follow-up so it gets its own
results.csv/config.json/data dir, and `--variants native` alone once
`n_objects=3` had already firmly established the stretch trick does
nothing for this model (no need to re-spend on stretched every time this
script is reused).

Self-contained (generates stimulus, sends to model, scores), mirroring
today's pylyshyn `run_pylyshyn_speed_sweep*.py` convention rather than the
older two-step generate-samples-then-diagnose flow
(`generate_openrouter_samples.py` + `run_openrouter_diagnostic.py`) --
question-building is reused from the latter via import (prompt text stays
in scripts/, per this project's convention, just shared across two debug_circular
scripts the same way pylyshyn's sweep scripts share `build_question` from
`run_pylyshyn_trial.py`).

Since `finst_video_model.debug_circular.stimulus_gen.generate_stimulus`
doesn't return the frame array directly (unlike pylyshyn's version), both
variants are produced by writing `video.npy` once (`save_mp4=False`) and
then calling `write_mp4` twice at different fps -- same technique as
`generate_openrouter_samples.py`.

N_SEEDS=20 (10 matching + 10 non-matching), same convention as pylyshyn's
sweeps.

Every trial's artifacts land in data/debug_circular/<run-name>/trials/
(gitignored); results.csv/config.json (tracked) go to
results/debug_circular/<run-name>/. results.csv is written incrementally
and re-running resumes automatically -- the resume key includes
`n_objects`, so re-running this script with a different `--n-objects`
under the same `--run-name` correctly adds new rows rather than being
mistaken for already-done work.

Usage:
    uv run python scripts/debug_circular/run_debug_circular_angle_sweep.py
    uv run python scripts/debug_circular/run_debug_circular_angle_sweep.py \\
        --run-name angle_sweep_n4 --n-objects 4 --variants native \\
        --rotation-degs 45 90 135 180 225 270 315 360
"""

import argparse
import concurrent.futures
import csv
import json
import os

import numpy as np
import requests

from finst_video_model.debug_circular.config import TrialConfig
from finst_video_model.debug_circular.stimulus_gen import generate_stimulus, write_mp4
from finst_video_model.scoring import classify_trial, compute_d_prime, parse_boolean_answer
from finst_video_model.vlm_client import ask_about_video
from run_openrouter_diagnostic import build_question

DEFAULT_MODEL = "google/gemma-4-31b-it"
REASONING = {"enabled": False}
N_SEEDS = 20
MAX_TOKENS = 100

DEFAULT_N_OBJECTS = 3
DEFAULT_ROTATION_DEGS = [40, 80, 120, 160, 200, 240, 280, 320, 360]
DEFAULT_VARIANTS = ["native", "stretched"]
DEFAULT_RUN_NAME = "angle_sweep"

STRETCH_TARGET_DURATION_S = 50.0

RESULT_FIELDS = [
    "trial_id", "model", "rotation_deg", "variant", "n_objects",
    "seed", "probe_on_target", "probe_is_target", "response", "predicted", "outcome",
    "prompt_tokens", "completion_tokens", "reasoning_tokens", "cost", "error",
]


def run_one_trial(
    batch_dir: str, model: str, n_objects: int, rotation_deg: float, variant: str,
    seed: int, probe_on_target: bool, stretch_encode_fps: float,
) -> dict:
    cfg = TrialConfig(n_objects=n_objects, rotation_deg=rotation_deg, probe_on_target=probe_on_target, seed=seed)
    trial_dir = os.path.join(batch_dir, "trials", cfg.trial_id)
    os.makedirs(trial_dir, exist_ok=True)

    ground_truth = generate_stimulus(cfg, trial_dir, save_mp4=False)
    frames = np.load(ground_truth["npy_path"])
    if variant == "stretched":
        video_path = os.path.join(trial_dir, "video_stretched.mp4")
        write_mp4(frames, video_path, fps=stretch_encode_fps)
    else:
        video_path = os.path.join(trial_dir, "video_native.mp4")
        write_mp4(frames, video_path, fps=cfg.fps)

    question = build_question(cfg.n_objects)
    with open(os.path.join(trial_dir, "question.txt"), "w") as f:
        f.write(question)

    row = {
        "trial_id": cfg.trial_id, "model": model, "rotation_deg": rotation_deg,
        "variant": variant, "n_objects": n_objects,
        "seed": seed, "probe_on_target": probe_on_target,
        "probe_is_target": ground_truth["probe_is_target"],
        "response": "", "predicted": "", "outcome": "",
        "prompt_tokens": "", "completion_tokens": "", "reasoning_tokens": "",
        "cost": "", "error": "",
    }
    result = dict(row, video_path=video_path, question=question)

    try:
        response_text, usage = ask_about_video(
            video_path, question, model,
            reasoning=REASONING, max_tokens=MAX_TOKENS,
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


def build_grid(rotation_degs: list[float], variants: list[str], seeds: list[int]):
    if len(seeds) % 2 != 0:
        raise ValueError(f"seeds must split evenly into matching/non-matching halves, got {len(seeds)}")
    half = len(seeds) // 2
    seeds_sorted = sorted(seeds)
    probe_on_target_by_seed = {seed: (i < half) for i, seed in enumerate(seeds_sorted)}
    grid = []
    for rotation_deg in rotation_degs:
        for variant in variants:
            for seed in seeds_sorted:
                grid.append((rotation_deg, variant, seed, probe_on_target_by_seed[seed]))
    return grid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", type=str, default=DEFAULT_RUN_NAME)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--n-objects", type=int, default=DEFAULT_N_OBJECTS)
    parser.add_argument("--rotation-degs", type=float, nargs="+", default=DEFAULT_ROTATION_DEGS)
    parser.add_argument("--variants", type=str, nargs="+", default=DEFAULT_VARIANTS, choices=["native", "stretched"])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(N_SEEDS)))
    parser.add_argument("--concurrency", type=int, default=6)
    args = parser.parse_args()

    native_total_frames = TrialConfig(n_objects=args.n_objects).total_frames
    native_fps = TrialConfig(n_objects=args.n_objects).fps
    stretch_encode_fps = native_total_frames / STRETCH_TARGET_DURATION_S

    print(
        f"n_objects={args.n_objects}. Native clip: {native_total_frames} frames @ {native_fps}fps "
        f"(~{native_total_frames / native_fps:.1f}s). "
        f"Stretched encode: {stretch_encode_fps:.2f}fps (~{STRETCH_TARGET_DURATION_S:.0f}s nominal)."
    )

    batch_dir = os.path.join("data", "debug_circular", args.run_name)
    results_dir = os.path.join("results", "debug_circular", args.run_name)
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
                "run_name": args.run_name, "arm": "debug_circular", "model": args.model,
                "reasoning": REASONING, "n_objects": args.n_objects,
                "rotation_degs": args.rotation_degs, "variants": args.variants,
                "stretch_encode_fps": stretch_encode_fps, "seeds": args.seeds,
            }, f, indent=2)

    results_csv = os.path.join(results_dir, "results.csv")
    already_ran = set()
    file_exists = os.path.isfile(results_csv)
    if file_exists:
        with open(results_csv) as f:
            for r in csv.DictReader(f):
                already_ran.add((
                    r["model"], int(r["n_objects"]), float(r["rotation_deg"]), r["variant"],
                    int(r["seed"]), r["probe_on_target"] == "True",
                ))

    grid = build_grid(args.rotation_degs, args.variants, args.seeds)
    pending = [spec for spec in grid if (args.model, args.n_objects, *spec) not in already_ran]
    total = len(grid)
    done = len(already_ran)

    with open(results_csv, "a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=RESULT_FIELDS)
        if not file_exists:
            writer.writeheader()
            csv_file.flush()

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {
                pool.submit(
                    run_one_trial, batch_dir, args.model, args.n_objects, rotation_deg, variant,
                    seed, probe_on_target, stretch_encode_fps,
                ): (rotation_deg, variant, seed, probe_on_target)
                for rotation_deg, variant, seed, probe_on_target in pending
            }
            for future in concurrent.futures.as_completed(futures):
                rotation_deg, variant, seed, probe_on_target = futures[future]
                row = future.result()
                done += 1
                print(
                    f"[{done}/{total}] rotation_deg={rotation_deg} variant={variant} "
                    f"seed={seed} probe_on_target={probe_on_target} -> {row['outcome'] or row['error']}"
                )
                writer.writerow(row)
                csv_file.flush()

    print(f"\nresults.csv up to date at {results_csv}")

    with open(results_csv) as f:
        all_rows = list(csv.DictReader(f))
    for rotation_deg in args.rotation_degs:
        for variant in args.variants:
            rows = [
                {"probe_is_target": r["probe_is_target"] == "True", "predicted": (
                    None if r["predicted"] == "" else r["predicted"] == "True"
                )}
                for r in all_rows
                if int(r["n_objects"]) == args.n_objects
                and float(r["rotation_deg"]) == rotation_deg and r["variant"] == variant
            ]
            if not rows:
                continue
            stats = compute_d_prime(rows)
            print(
                f"rotation_deg={rotation_deg} variant={variant}: "
                f"d'={stats['d_prime']:.2f} (n={len(rows)}, unparseable={stats['n_unparseable']})"
            )


if __name__ == "__main__":
    main()
