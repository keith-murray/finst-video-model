"""
Batch runner for claude/2026_08/2026_08_26/TODO.md's "Part 3": now that the
duration-stretched mp4 trick (scripts/debug_circular/run_openrouter_diagnostic.py,
results/debug_circular/openrouter_diagnostic/) is confirmed to get OpenRouter
to retain far more video frames, revisit the pylyshyn stimulus -- ported to
debug_circular's visual design (see finst_video_model.pylyshyn.config's
module docstring) -- with a 3-axis sweep: model x reasoning_effort x
video variant (native vs. stretched), at the field size we're starting with
(N_OBJECTS=3, N_CUED=1).

Unlike run_pylyshyn_batch.py/run_pylyshyn_reasoning_batch.py (which
regenerate a trial's video fresh per grid point even though content only
depends on (n_cued, seed, probe_on_target)), stimulus generation here is
still done once per grid point too -- content is a cheap deterministic
function of (seed, probe_on_target), so re-deriving it is fast, and (unlike
debug_circular's cluster pipeline) nothing here persists a raw .npy to
disk, so the repeated small-mp4 writes cost trivial disk. The returned
in-memory frame array (finst_video_model.pylyshyn.stimulus_gen.generate_stimulus
now returns `(ground_truth, frames)`) is what makes the "stretched" variant
re-encode lossless -- see finst_video_model.debug_circular.stimulus_gen.write_mp4,
reused here since it's a generic frames-array-to-mp4 encoder with no
debug_circular-specific logic.

REASONING_LEVELS_BY_MODEL sweeps {none, low, medium} for qwen3.8-27b
(reasoning optional there) but only {low, medium} for qwen3.8-max --
qwen3.8-max's reasoning is mandatory (confirmed live in
run_pylyshyn_reasoning_batch.py: disabling 400s with "Reasoning is
mandatory for this endpoint and cannot be disabled"), so a "none" point
isn't requestable for it, matching that script's precedent.

STRETCH_ENCODE_FPS targets the same ~50s nominal duration that worked for
debug_circular (see [[reference-openrouter-duration-stretch-trick]] in
memory) rather than reusing debug_circular's literal fps=2 -- this
stimulus's native clip is much longer (18s vs. debug_circular's 10s), so
matching the same *target duration* (not the same encode fps) is the
better analogy.

8 seeds per (model, reasoning_level, variant) condition, split evenly
matching/non-matching (probe_on_target), same convention as
run_pylyshyn_reasoning_batch.py -- 16 trials/condition.

Every trial's artifacts land in data/pylyshyn/stretch_sweep/trials/<trial_id>/
(gitignored); results.csv/config.json (tracked) go to
results/pylyshyn/reasoning_sweeps/stretch_sweep/. results.csv is written incrementally and
re-running resumes automatically (already-run (model, reasoning_level,
variant, seed, probe_on_target) combinations are skipped).

Usage:
    uv run python scripts/pylyshyn/reasoning_sweeps/run_pylyshyn_stretch_sweep.py
    uv run python scripts/pylyshyn/reasoning_sweeps/run_pylyshyn_stretch_sweep.py --models qwen/qwen3.8-27b
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

N_OBJECTS = 3
N_CUED = 1
N_SEEDS = 16

# Bounds total response length. Investigated 2026-08-26: does NOT raise
# qwen3.8-27b's low/medium reasoning-effort budget, which is pinned at a
# small, reproducible value (~65/~257 reasoning tokens) regardless of this
# setting -- that's a real property of this model's reasoning.effort tiers
# on /chat/completions, not truncation. Kept for its ordinary purpose
# (bounding response length) -- see finst_video_model.vlm_client
# .ask_about_video's docstring for the full investigation.
MAX_TOKENS = 8192

REASONING_LEVELS_BY_MODEL = {
    "qwen/qwen3.8-27b": ["none", "low", "medium", "high"],
    "qwen/qwen3.8-max": ["low", "medium"],  # reasoning is mandatory, no "none"
    # Part 4 (claude/2026_08/2026_08_26/TODO.md): can a non-reasoning model
    # solve this purely feedforward? reasoning.mandatory=false per the live
    # OpenRouter /models listing (checked 2026-08-26), so "none" is valid.
    "qwen/qwen3.5-122b-a10b": ["none"],  # only 10B active params (MoE) -- user flagged this after the fact
    "qwen/qwen3.6-plus": ["none"],  # larger, denser follow-up per the same question
}
VARIANTS = ["native", "stretched"]

RUN_NAME = "stretch_sweep"

STRETCH_TARGET_DURATION_S = 50.0
_NATIVE_TOTAL_FRAMES = TrialConfig(n_objects=N_OBJECTS, n_cued=N_CUED).total_frames
STRETCH_ENCODE_FPS = _NATIVE_TOTAL_FRAMES / STRETCH_TARGET_DURATION_S

RESULT_FIELDS = [
    "trial_id", "model", "reasoning_level", "variant", "n_objects", "n_cued",
    "seed", "probe_on_target", "probe_is_target", "response", "predicted", "outcome",
    "prompt_tokens", "completion_tokens", "reasoning_tokens", "cost", "error",
]


def reasoning_param(level: str) -> dict:
    return {"enabled": False} if level == "none" else {"effort": level}


def run_one_trial(
    batch_dir: str, model: str, reasoning_level: str, variant: str,
    seed: int, probe_on_target: bool,
) -> dict:
    cfg = TrialConfig(n_objects=N_OBJECTS, n_cued=N_CUED, probe_on_target=probe_on_target, seed=seed)
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
        "trial_id": cfg.trial_id, "model": model, "reasoning_level": reasoning_level,
        "variant": variant, "n_objects": N_OBJECTS, "n_cued": N_CUED,
        "seed": seed, "probe_on_target": probe_on_target,
        "probe_is_target": ground_truth["probe_is_target"],
        "video_path": video_path, "question": question,
    }
    row = {
        "trial_id": cfg.trial_id, "model": model, "reasoning_level": reasoning_level,
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
            reasoning=reasoning_param(reasoning_level), max_tokens=MAX_TOKENS,
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
        for level in REASONING_LEVELS_BY_MODEL[model]:
            for variant in VARIANTS:
                for seed in seeds_sorted:
                    grid.append((model, level, variant, seed, probe_on_target_by_seed[seed]))
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

    print(
        f"Native clip: {_NATIVE_TOTAL_FRAMES} frames @ {TrialConfig().fps}fps "
        f"(~{_NATIVE_TOTAL_FRAMES / TrialConfig().fps:.1f}s). "
        f"Stretched encode: {STRETCH_ENCODE_FPS:.2f}fps (~{STRETCH_TARGET_DURATION_S:.0f}s nominal)."
    )

    batch_dir = os.path.join("data", "pylyshyn", RUN_NAME)
    results_dir = os.path.join("results", "pylyshyn", "reasoning_sweeps", RUN_NAME)
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
                "reasoning_levels_by_model": REASONING_LEVELS_BY_MODEL,
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
                    r["model"], r["reasoning_level"], r["variant"],
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
                pool.submit(run_one_trial, batch_dir, model, level, variant, seed, probe_on_target): (
                    model, level, variant, seed, probe_on_target
                )
                for model, level, variant, seed, probe_on_target in pending
            }
            for future in concurrent.futures.as_completed(futures):
                model, level, variant, seed, probe_on_target = futures[future]
                row = future.result()
                done += 1
                print(
                    f"[{done}/{total}] model={model} reasoning_level={level} variant={variant} "
                    f"seed={seed} probe_on_target={probe_on_target} -> {row['outcome'] or row['error']}"
                )
                writer.writerow(row)
                csv_file.flush()

    print(f"\nresults.csv up to date at {results_csv}")

    with open(results_csv) as f:
        all_rows = list(csv.DictReader(f))
    for model in args.models:
        for level in REASONING_LEVELS_BY_MODEL[model]:
            for variant in VARIANTS:
                rows = [
                    {"probe_is_target": r["probe_is_target"] == "True", "predicted": (
                        None if r["predicted"] == "" else r["predicted"] == "True"
                    )}
                    for r in all_rows
                    if r["model"] == model and r["reasoning_level"] == level and r["variant"] == variant
                ]
                if not rows:
                    continue
                stats = compute_d_prime(rows)
                print(
                    f"model={model} level={level} variant={variant}: "
                    f"d'={stats['d_prime']:.2f} (n={len(rows)}, unparseable={stats['n_unparseable']})"
                )


if __name__ == "__main__":
    main()
