"""
Sends debug_circular sample videos to an OpenRouter-hosted model to test
whether the artificially-stretched-duration trick
(scripts/debug_circular/generate_openrouter_samples.py, see
claude/2026_08/2026_08_26/TODO.md's "Part 2") actually changes how many
frames the model gets to see, compared to the same content's native-fps
encoding.

Sends BOTH variants of each condition -- "native" (data/debug_circular/
debug_circular_samples/, fps=10, ~10s) and "stretched" (data/debug_circular/
openrouter_stretch_samples/, fps=2, ~50s nominal) -- since both directories
were generated with the identical rotation_deg/clockwise/probe_on_target/
seed sweep formula, so pixel content is identical between a native/stretched
pair; only the mp4 container's declared fps/duration differs. Two signals
distinguish "did the model actually get more frames":
  1. `prompt_tokens` from the API response's usage dict -- for multimodal
     input this should scale with how many video frames the provider's
     preprocessing actually retained, independent of the model's reasoning
     or answer quality.
  2. Task accuracy (d' via finst_video_model.scoring) -- a video that's
     been downsampled to only a couple of frames makes the rigid-ring
     rotation task fundamentally ambiguous (can't tell direction/magnitude
     from 2 frames alone), so accuracy is an indirect but meaningful proxy
     too.

Question-building is duplicated here (not imported from
run_cluster_batch.py) per this project's convention that prompt-building
code lives only in scripts/, never shared library code.

Every trial's raw artifacts land in
data/debug_circular/openrouter_diagnostic/trials/<video_variant>_<trial_id>/
(gitignored); results.csv (tracked) goes to
results/debug_circular/openrouter_diagnostic/. results.csv is written
incrementally and a re-run resumes automatically (already-run
(video_variant, trial_id, model, reasoning_level) combinations are
skipped -- so re-running with a different --model/--reasoning-effort adds a
new set of rows for the same trials rather than skipping them).

Usage:
    uv run python scripts/debug_circular/run_openrouter_diagnostic.py
    uv run python scripts/debug_circular/run_openrouter_diagnostic.py --reasoning-effort none
"""

import argparse
import concurrent.futures
import csv
import json
import os

import requests

from finst_video_model.scoring import classify_trial, compute_d_prime, parse_boolean_answer
from finst_video_model.vlm_client import ask_about_video

RUN_NAME = "openrouter_diagnostic"

RESULT_FIELDS = [
    "trial_id", "video_variant", "model", "reasoning_level",
    "n_objects", "rotation_deg", "clockwise", "probe_on_target", "seed",
    "probe_is_target", "response", "predicted", "outcome",
    "prompt_tokens", "completion_tokens", "reasoning_tokens", "cost", "error",
]


def build_question(n_objects: int) -> str:
    return (
        f"You will watch a video of {n_objects} white crosses (+) arranged "
        f"equally spaced around a circle on a black background. At the "
        f"start of the video, all crosses are stationary, and one of them "
        f"-- the cued cross -- is colored red, while the rest stay white. "
        f"After a moment it turns white too, and all crosses then rotate "
        f"together, as a single rigid group, around the circle at a "
        f"constant speed and direction, maintaining their equal spacing "
        f"the whole time -- there is no more color difference between "
        f"crosses once they start moving, so you can only keep track of "
        f"which cross is which by its position in the rotating group. "
        f"Near the end of the video, all crosses stop moving, and exactly "
        f"one of them turns red.\n\n"
        f"Question: was the cross that turned red at the end the same "
        f"physical cross that was red at the very start of the video?\n\n"
        f"Answer with only the single word True or False. Do not include "
        f"any other text in your answer."
    )


def reasoning_param(level: str) -> dict:
    return {"enabled": False} if level == "none" else {"effort": level}


def discover_trials(root: str) -> list[str]:
    return sorted(
        d for d in os.listdir(root)
        if os.path.isfile(os.path.join(root, d, "video.mp4"))
        and os.path.isfile(os.path.join(root, d, "ground_truth.json"))
    )


def run_one_trial(
    data_dir: str, video_variant: str, trial_dir: str, trial_id: str,
    model: str, reasoning_level: str,
) -> dict:
    with open(os.path.join(trial_dir, "ground_truth.json")) as f:
        ground_truth = json.load(f)
    cfg = ground_truth["config"]
    video_path = os.path.join(trial_dir, "video.mp4")
    question = build_question(cfg["n_objects"])

    out_dir = os.path.join(data_dir, "trials", f"{video_variant}_{trial_id}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "question.txt"), "w") as f:
        f.write(question)

    row = {
        "trial_id": trial_id, "video_variant": video_variant, "model": model,
        "reasoning_level": reasoning_level, "n_objects": cfg["n_objects"],
        "rotation_deg": cfg["rotation_deg"], "clockwise": cfg["clockwise"],
        "probe_on_target": cfg["probe_on_target"], "seed": cfg["seed"],
        "probe_is_target": ground_truth["probe_is_target"],
        "response": "", "predicted": "", "outcome": "",
        "prompt_tokens": "", "completion_tokens": "", "reasoning_tokens": "",
        "cost": "", "error": "",
    }
    result = dict(row, video_path=video_path, question=question)

    try:
        response_text, usage = ask_about_video(
            video_path, question, model,
            reasoning=reasoning_param(reasoning_level), return_usage=True,
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

    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)

    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-root", type=str, default="data/debug_circular/debug_circular_samples")
    parser.add_argument("--stretched-root", type=str, default="data/debug_circular/openrouter_stretch_samples")
    parser.add_argument("--model", type=str, default="qwen/qwen3.8-27b")
    parser.add_argument(
        "--reasoning-effort", type=str, default="low",
        choices=["none", "minimal", "low", "medium", "high"],
    )
    parser.add_argument("--concurrency", type=int, default=5)
    args = parser.parse_args()

    data_dir = os.path.join("data", "debug_circular", RUN_NAME)
    results_dir = os.path.join("results", "debug_circular", RUN_NAME)
    os.makedirs(os.path.join(data_dir, "trials"), exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    roots = {"native": args.native_root, "stretched": args.stretched_root}
    grid = [
        (video_variant, os.path.join(root, trial_id), trial_id)
        for video_variant, root in roots.items()
        for trial_id in discover_trials(root)
    ]
    print(f"Discovered {sum(1 for v, r in roots.items() for _ in discover_trials(r))} trials "
          f"across {list(roots.keys())}")

    results_csv = os.path.join(results_dir, "results.csv")
    already_ran = set()
    file_exists = os.path.isfile(results_csv)
    if file_exists:
        with open(results_csv) as f:
            for r in csv.DictReader(f):
                already_ran.add((r["video_variant"], r["trial_id"], r["model"], r["reasoning_level"]))

    pending = [
        spec for spec in grid
        if (spec[0], spec[2], args.model, args.reasoning_effort) not in already_ran
    ]
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
                    run_one_trial, data_dir, video_variant, trial_dir, trial_id,
                    args.model, args.reasoning_effort,
                ): (video_variant, trial_id)
                for video_variant, trial_dir, trial_id in pending
            }
            for future in concurrent.futures.as_completed(futures):
                video_variant, trial_id = futures[future]
                row = future.result()
                done += 1
                print(
                    f"[{done}/{total}] variant={video_variant} trial={trial_id} "
                    f"prompt_tokens={row['prompt_tokens']} -> {row['outcome'] or row['error']}"
                )
                writer.writerow(row)
                csv_file.flush()

    print(f"\nresults.csv up to date at {results_csv}")

    with open(results_csv) as f:
        all_rows = list(csv.DictReader(f))
    print(f"\n=== Summary for model={args.model} reasoning_level={args.reasoning_effort} ===")
    for video_variant in roots:
        rows = [
            r for r in all_rows
            if r["video_variant"] == video_variant
            and r["model"] == args.model and r["reasoning_level"] == args.reasoning_effort
        ]
        token_vals = [int(r["prompt_tokens"]) for r in rows if r["prompt_tokens"]]
        mean_tokens = sum(token_vals) / len(token_vals) if token_vals else float("nan")
        scorable = [
            {"probe_is_target": r["probe_is_target"] == "True", "predicted": (
                None if r["predicted"] == "" else r["predicted"] == "True"
            )}
            for r in rows
        ]
        stats = compute_d_prime(scorable)
        print(
            f"\n--- {video_variant} (n={len(rows)}) ---\n"
            f"  mean prompt_tokens: {mean_tokens:.0f}\n"
            f"  d'={stats['d_prime']:.2f} (hits={stats['hits']}, misses={stats['misses']}, "
            f"false_alarms={stats['false_alarms']}, correct_rejections={stats['correct_rejections']}, "
            f"unparseable={stats['n_unparseable']})"
        )


if __name__ == "__main__":
    main()
