"""
Batch runner for claude/2026_08/2026_08_27/TODO.md's "Task 3": generalizes
Task 2's redirect x speed x variant sweep (run_pylyshyn_speed_sweep.py,
results/pylyshyn/speed_sweep/) to test other models, at a bumped-up 20
seeds/condition (10 matching + 10 non-matching, was 16/8+8).

Lives in its own results/pylyshyn/speed_sweep_models/ +
data/pylyshyn/speed_sweep_models/ directories rather than extending
Task 2's results.csv, since that script hardcodes a single model and
locks a single shared 16-seed set in config.json -- changing either
would retroactively touch Task 2's already-analyzed qwen3.6-plus data.

MODEL_REASONING: the TODO asked for `z-ai/glm-5.3-flash` "with no
reasoning", but its live OpenRouter listing (checked 2026-08-27) shows
`reasoning.mandatory=true` (supported_efforts max/high/low, no disable) --
same situation as qwen3.8-max. Per the user, substituted
`z-ai/glm-5v-turbo` (same vendor family, `reasoning.mandatory=false`)
instead. Kept as a dict (not a single hardcoded model) so more models can
be added the way the TODO's "we can even try this on other models" implies,
each with its own reasoning param (mirrors
run_pylyshyn_stretch_sweep.py's REASONING_LEVELS_BY_MODEL convention).

Every trial's artifacts land in
data/pylyshyn/speed_sweep_models/trials/<trial_id>/ (gitignored);
results.csv/config.json (tracked) go to
results/pylyshyn/speed_sweep_models/. results.csv is written incrementally
and re-running resumes automatically (already-run (model,
redirect_condition, speed_condition, variant, seed, probe_on_target)
combinations are skipped).

Usage:
    uv run python scripts/pylyshyn/run_pylyshyn_speed_sweep_models.py
    uv run python scripts/pylyshyn/run_pylyshyn_speed_sweep_models.py --models z-ai/glm-5v-turbo
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
from run_pylyshyn_trial import build_question

N_OBJECTS = 3
N_CUED = 1
N_SEEDS = 20
MAX_TOKENS = 8192

# See module docstring: z-ai/glm-5.3-flash (the TODO's original ask) has
# mandatory reasoning on OpenRouter, so no "disabled" point exists for it --
# substituted z-ai/glm-5v-turbo (reasoning.mandatory=false) per the user.
MODEL_REASONING = {
    "z-ai/glm-5v-turbo": {"enabled": False},
}

REDIRECT_CONDITIONS = {"slow": 2.0, "fast": 1.0}
SPEED_CONDITIONS = {"slow": (16.0, 33.0), "fast": (32.0, 66.0)}
VARIANTS = ["native", "stretched"]

RUN_NAME = "speed_sweep_models"

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
    batch_dir: str, model: str, redirect_condition: str, speed_condition: str, variant: str,
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
        "trial_id": cfg.trial_id, "model": model,
        "redirect_condition": redirect_condition, "speed_condition": speed_condition,
        "variant": variant, "n_objects": N_OBJECTS, "n_cued": N_CUED,
        "seed": seed, "probe_on_target": probe_on_target,
        "probe_is_target": ground_truth["probe_is_target"],
        "video_path": video_path, "question": question,
    }
    row = {
        "trial_id": cfg.trial_id, "model": model,
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
            video_path, question, model,
            reasoning=MODEL_REASONING[model], max_tokens=MAX_TOKENS,
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


def build_grid(models: list[str], seeds: list[int]):
    if len(seeds) % 2 != 0:
        raise ValueError(f"seeds must split evenly into matching/non-matching halves, got {len(seeds)}")
    half = len(seeds) // 2
    seeds_sorted = sorted(seeds)
    probe_on_target_by_seed = {seed: (i < half) for i, seed in enumerate(seeds_sorted)}
    grid = []
    for model in models:
        for redirect_condition in REDIRECT_CONDITIONS:
            for speed_condition in SPEED_CONDITIONS:
                for variant in VARIANTS:
                    for seed in seeds_sorted:
                        grid.append((model, redirect_condition, speed_condition, variant, seed, probe_on_target_by_seed[seed]))
    return grid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=str, nargs="+", default=list(MODEL_REASONING.keys()))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(N_SEEDS)))
    parser.add_argument("--concurrency", type=int, default=5)
    args = parser.parse_args()

    for model in args.models:
        if model not in MODEL_REASONING:
            raise ValueError(f"No reasoning param for model {model!r}; add it to MODEL_REASONING")

    print(
        f"Native clip: {_NATIVE_TOTAL_FRAMES} frames @ {TrialConfig().fps}fps "
        f"(~{_NATIVE_TOTAL_FRAMES / TrialConfig().fps:.1f}s). "
        f"Stretched encode: {STRETCH_ENCODE_FPS:.2f}fps (~{STRETCH_TARGET_DURATION_S:.0f}s nominal)."
    )

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
                "run_name": RUN_NAME, "arm": "pylyshyn", "models": args.models,
                "n_objects": N_OBJECTS, "n_cued": N_CUED,
                "model_reasoning": MODEL_REASONING,
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
                    r["model"], r["redirect_condition"], r["speed_condition"], r["variant"],
                    int(r["seed"]), r["probe_on_target"] == "True",
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
                pool.submit(run_one_trial, batch_dir, model, redirect_condition, speed_condition, variant, seed, probe_on_target): (
                    model, redirect_condition, speed_condition, variant, seed, probe_on_target
                )
                for model, redirect_condition, speed_condition, variant, seed, probe_on_target in pending
            }
            for future in concurrent.futures.as_completed(futures):
                model, redirect_condition, speed_condition, variant, seed, probe_on_target = futures[future]
                row = future.result()
                done += 1
                print(
                    f"[{done}/{total}] model={model} redirect={redirect_condition} speed={speed_condition} "
                    f"variant={variant} seed={seed} probe_on_target={probe_on_target} -> {row['outcome'] or row['error']}"
                )
                writer.writerow(row)
                csv_file.flush()

    print(f"\nresults.csv up to date at {results_csv}")

    with open(results_csv) as f:
        all_rows = list(csv.DictReader(f))
    for model in args.models:
        for redirect_condition in REDIRECT_CONDITIONS:
            for speed_condition in SPEED_CONDITIONS:
                for variant in VARIANTS:
                    rows = [
                        {"probe_is_target": r["probe_is_target"] == "True", "predicted": (
                            None if r["predicted"] == "" else r["predicted"] == "True"
                        )}
                        for r in all_rows
                        if r["model"] == model and r["redirect_condition"] == redirect_condition
                        and r["speed_condition"] == speed_condition and r["variant"] == variant
                    ]
                    if not rows:
                        continue
                    stats = compute_d_prime(rows)
                    print(
                        f"model={model} redirect={redirect_condition} speed={speed_condition} variant={variant}: "
                        f"d'={stats['d_prime']:.2f} (n={len(rows)}, unparseable={stats['n_unparseable']})"
                    )


if __name__ == "__main__":
    main()
