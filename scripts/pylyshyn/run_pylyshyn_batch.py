"""
Batch runner for the pylyshyn arm's n_objects x n_cued x model grid
(claude/2026_08_19/TODO.md's "Testing Qwen 3.8" section).

Grid: n_objects in {2,4,6,8,10}; per n_objects, n_cued sweeps 1,3,5,...
stopping once a value would exceed half of n_objects (see
`default_n_cued_values`) -- giving conditions (2:[1], 4:[1], 6:[1,3],
8:[1,3], 10:[1,3,5]), 9 conditions total. Each condition gets N_SEEDS seeds
split evenly: the first half probed on-target (matching, for the hit rate),
the second half on-distractor (non-matching, for the false-alarm rate) --
20 seeds -> 20 trials/condition, as needed by
finst_video_model.comprehension.scoring.compute_d_prime. This is a distinct
seed per trial, unlike the earlier fd799482/e97a6d18 batches (see
scripts/prototype), which ran each seed twice (paired design, 2x the
trials/condition).

Models default to both Qwen 3.8 variants (see MODEL_CONFIGS for pinned
reasoning settings), each with its own row in results.csv (a `model` column)
so accuracy/d' can be compared side by side. qwen3.8-max has mandatory
reasoning (its "minimal" floor still burns real tokens, ~$0.031/trial
measured during cost estimation) -- unlike Gemini there's no cheaper
alternate provider worth pinning (checked via OpenRouter's
/models/{id}/endpoints; only marginal, <15% differences across providers).
qwen3.8-27b's reasoning is optional, so it was first tried fully disabled
(`reasoning: {"enabled": False}`, ~$0.005/trial) -- cheap and reliable (no
risk of the completion budget going entirely to reasoning tokens and
returning no answer content at all, an unhandled `None` that crashed an
earlier run of this script; ask_about_video now raises a catchable
RuntimeError for that case instead), but an 18-trial spot check
(results/pylyshyn/qwen3.8-27b-low/, vs. the disabled-reasoning
results/pylyshyn/qwen3.8-27b/ on the same trials) found d' swings from
-1.71 (disabled) to +0.77 ("low") -- disabling reasoning entirely was
crippling the model, not just saving money. MODEL_CONFIGS now pins
qwen3.8-27b to `{"effort": "low"}` to match.

Every trial's artifacts land in data/pylyshyn/<run_name>/trials/<trial_id>/
(video, ground_truth, question, result -- data/ is gitignored); results.csv
and config.json (small, tracked) go to results/pylyshyn/<run_name>/.
`run_name` defaults to a name derived from the model(s) being run (e.g.
"qwen3.8-27b") but can be overridden with --run-name -- needed whenever a
model is re-run under a different (non-default-key) setting, such as a
reasoning-effort comparison, so the new run gets its own directory instead
of silently no-op'ing against the old run's results.csv (dedup keys on
(model, n_objects, n_cued, seed, probe_on_target), not on reasoning
config, so same-model-different-config runs must be separated by
--run-name, not left to collide). results.csv is written incrementally,
and simply re-running the same --models/--n-objects/--seeds/--run-name
resumes automatically -- already-completed combinations are skipped, not
re-run/re-paid-for.

Usage:
    uv run python scripts/pylyshyn/run_pylyshyn_batch.py
    uv run python scripts/pylyshyn/run_pylyshyn_batch.py --models qwen/qwen3.8-27b
    uv run python scripts/pylyshyn/run_pylyshyn_batch.py --models qwen/qwen3.8-27b --run-name qwen3.8-27b-low
"""

import argparse
import concurrent.futures
import csv
import json
import os

import requests

from finst_video_model.comprehension.pylyshyn.config import TrialConfig
from finst_video_model.comprehension.pylyshyn.stimulus_gen import generate_stimulus
from finst_video_model.comprehension.scoring import classify_trial, parse_boolean_answer
from finst_video_model.comprehension.vlm_client import ask_about_video
from run_pylyshyn_trial import build_question

MODEL_CONFIGS = {
    "qwen/qwen3.8-27b": {"reasoning": {"effort": "low"}},
    "qwen/qwen3.8-max": {"reasoning": {"effort": "minimal"}},
}
N_OBJECTS_DEFAULT = [2, 4, 6, 8, 10]
N_SEEDS = 20

RESULT_FIELDS = [
    "trial_id", "model", "n_objects", "n_cued", "seed", "probe_on_target",
    "probe_is_target", "response", "predicted", "outcome", "cost", "error",
]


def default_n_cued_values(n_objects: int) -> list[int]:
    """1, 3, 5, ... stopping once a value would exceed half of n_objects --
    matches claude/2026_08_19/TODO.md's rule exactly (e.g. 10 -> [1, 3, 5])."""
    values = []
    cue = 1
    while cue <= n_objects / 2:
        values.append(cue)
        cue += 2
    return values


def run_one_trial(
    batch_dir: str, model: str, reasoning: dict,
    n_objects: int, n_cued: int, seed: int, probe_on_target: bool,
) -> dict:
    cfg = TrialConfig(n_objects=n_objects, n_cued=n_cued, probe_on_target=probe_on_target, seed=seed)
    trial_dir = os.path.join(batch_dir, "trials", cfg.trial_id)
    os.makedirs(trial_dir, exist_ok=True)

    ground_truth = generate_stimulus(cfg, trial_dir)
    question = build_question(cfg)
    with open(os.path.join(trial_dir, "question.txt"), "w") as f:
        f.write(question)

    result = {
        "trial_id": cfg.trial_id, "model": model, "n_objects": n_objects, "n_cued": n_cued,
        "seed": seed, "probe_on_target": probe_on_target,
        "probe_is_target": ground_truth["probe_is_target"],
        "video_path": ground_truth["video_path"], "question": question,
    }
    row = {
        "trial_id": cfg.trial_id, "model": model, "n_objects": n_objects, "n_cued": n_cued,
        "seed": seed, "probe_on_target": probe_on_target,
        "probe_is_target": ground_truth["probe_is_target"],
        "response": "", "predicted": "", "outcome": "", "cost": "", "error": "",
    }

    try:
        response_text, usage = ask_about_video(
            ground_truth["video_path"], question, model,
            reasoning=reasoning, return_usage=True,
        )
        predicted = parse_boolean_answer(response_text)
        outcome = (
            classify_trial(ground_truth["probe_is_target"], predicted)
            if predicted is not None else "unparseable"
        )
        result.update({"response": response_text, "predicted": predicted, "outcome": outcome, "usage": usage})
        row.update({
            "response": response_text, "predicted": predicted, "outcome": outcome,
            "cost": usage.get("cost") if usage else "",
        })
    except (requests.RequestException, KeyError, RuntimeError) as e:
        result["error"] = str(e)
        row["error"] = str(e)

    with open(os.path.join(trial_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)

    return row


def run_name_for(models: list[str]) -> str:
    """Human-readable, deterministic run directory name derived from the
    model slug(s) being run (e.g. "qwen/qwen3.8-27b" -> "qwen3.8-27b"),
    joined with "_" for a multi-model run."""
    return "_".join(model.split("/")[-1] for model in models)


def build_grid(models: list[str], n_objects_list: list[int], seeds: list[int]):
    if len(seeds) % 2 != 0:
        raise ValueError(f"seeds must split evenly into matching/non-matching halves, got {len(seeds)}")
    half = len(seeds) // 2
    seeds_sorted = sorted(seeds)
    probe_on_target_by_seed = {
        seed: (i < half) for i, seed in enumerate(seeds_sorted)
    }
    grid = []
    for model in models:
        for n_objects in n_objects_list:
            for n_cued in default_n_cued_values(n_objects):
                for seed in seeds_sorted:
                    grid.append((model, n_objects, n_cued, seed, probe_on_target_by_seed[seed]))
    return grid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=str, nargs="+", default=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--n-objects", type=int, nargs="+", default=N_OBJECTS_DEFAULT)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(N_SEEDS)))
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument(
        "--run-name", type=str, default=None,
        help="Overrides the default model-derived run directory name -- required when "
             "re-running a model under a different (non-MODEL_CONFIGS-default) setting, "
             "so it doesn't collide with that model's default-config results.",
    )
    args = parser.parse_args()

    for model in args.models:
        if model not in MODEL_CONFIGS:
            raise ValueError(f"No reasoning config for model {model!r}; add it to MODEL_CONFIGS")

    run_name = args.run_name or run_name_for(args.models)
    batch_dir = os.path.join("data", "pylyshyn", run_name)
    results_dir = os.path.join("results", "pylyshyn", run_name)
    os.makedirs(os.path.join(batch_dir, "trials"), exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    config_path = os.path.join(results_dir, "config.json")
    if os.path.isfile(config_path):
        # build_grid's matching/non-matching split depends on a seed's
        # position within the *whole* --seeds list (sorted), so re-running
        # against a different seed set here would silently relabel some
        # already-collected seeds' probe_on_target going forward -- caught
        # this the hard way once already (a 2-seed smoke test left a
        # seed=1 row with a different label than the full 20-seed run gave
        # it, producing a stray 181st trial for one condition).
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
                "run_name": run_name, "arm": "pylyshyn", "models": args.models,
                "reasoning": {m: MODEL_CONFIGS[m]["reasoning"] for m in args.models},
                "n_objects": args.n_objects,
                "n_cued_by_n_objects": {n: default_n_cued_values(n) for n in args.n_objects},
                "seeds": args.seeds,
            }, f, indent=2)

    results_csv = os.path.join(results_dir, "results.csv")
    already_ran = set()
    file_exists = os.path.isfile(results_csv)
    if file_exists:
        with open(results_csv) as f:
            for r in csv.DictReader(f):
                already_ran.add((
                    r["model"], int(r["n_objects"]), int(r["n_cued"]),
                    int(r["seed"]), r["probe_on_target"] == "True",
                ))

    grid = build_grid(args.models, args.n_objects, args.seeds)
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
                pool.submit(
                    run_one_trial, batch_dir, model,
                    MODEL_CONFIGS[model]["reasoning"], n_objects, n_cued, seed, probe_on_target,
                ): (model, n_objects, n_cued, seed, probe_on_target)
                for model, n_objects, n_cued, seed, probe_on_target in pending
            }
            for future in concurrent.futures.as_completed(futures):
                model, n_objects, n_cued, seed, probe_on_target = futures[future]
                row = future.result()
                done += 1
                print(
                    f"[{done}/{total}] model={model} n_objects={n_objects} n_cued={n_cued} "
                    f"seed={seed} probe_on_target={probe_on_target} -> {row['outcome'] or row['error']}"
                )
                writer.writerow(row)
                csv_file.flush()

    print(f"\nresults.csv up to date at {results_csv}")
    print(f"run_name={run_name}")


if __name__ == "__main__":
    main()
