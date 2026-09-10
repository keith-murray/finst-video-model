"""
Aggregates raw cluster_response_nframes{N}_{sampling}.json outputs (written by
scripts/debug_circular/run_gemma_cluster_batch.py on the cluster, then rsynced
back down) into a scored results.csv, reusing this project's existing
True/False parsing and signal-detection scoring (finst_video_model.scoring)
rather than reimplementing it -- same approach as
scripts/debug_circular/aggregate_cluster_results.py.

Kept as a separate script rather than generalizing aggregate_cluster_results.py,
since num_frames/sampling are properties of the inference run (recorded only
in each cluster_response_nframes{N}_{sampling}.json), not of
TrialConfig/ground_truth.json the way rotation_deg is -- the existing
aggregator's "group by a config column" shape doesn't fit this sweep axis.

--response-prefix selects which cluster batch script's output to score:
"nframes" (default) for run_gemma_cluster_batch.py's do_sample_frames=True
auto-sampled output (cluster_response_nframes{N}_{sampling}.json), or
"manualframes" for run_gemma_cluster_batch_manual_frames.py's bypass-and-
manually-slice diagnostic (cluster_response_manualframes{N}_{sampling}.json)
-- see that script's docstring for why it exists. --out-root defaults to a
prefix-specific directory so the two pipelines' results.csv never collide.

Usage:
    uv run python scripts/debug_circular/aggregate_gemma_frame_sweep.py \\
        --num-frames 4 8 16 24 32 50 100 --sampling greedy recommended
    uv run python scripts/debug_circular/aggregate_gemma_frame_sweep.py \\
        --response-prefix manualframes --num-frames 4 8 16 24 32 50 100 --sampling greedy recommended
"""

import argparse
import csv
import json
import os

from finst_video_model.scoring import classify_trial, compute_d_prime, parse_boolean_answer

RESULT_FIELDS = [
    "trial_id", "model_tag", "n_objects", "rotation_deg", "clockwise",
    "probe_on_target", "seed", "num_frames", "sampling", "probe_is_target",
    "response", "predicted", "outcome", "prompt_tokens", "output_tokens",
    "generation_time_s", "error",
]


def load_trial_row(
    trial_dir: str, trial_id: str, num_frames: int, sampling: str, response_file: str,
) -> dict | None:
    gt_path = os.path.join(trial_dir, "ground_truth.json")
    response_path = os.path.join(trial_dir, response_file)
    if not os.path.isfile(gt_path):
        print(f"[{trial_id}, n_frames={num_frames}, sampling={sampling}] missing ground_truth.json, skipping")
        return None
    if not os.path.isfile(response_path):
        print(f"[{trial_id}, n_frames={num_frames}, sampling={sampling}] no {response_file} yet, skipping")
        return None

    with open(gt_path) as f:
        ground_truth = json.load(f)
    with open(response_path) as f:
        response = json.load(f)

    cfg = ground_truth["config"]
    response_text = response.get("response_text").get("content")
    predicted = parse_boolean_answer(response_text) if response_text else None
    if predicted is not None:
        outcome = classify_trial(ground_truth["probe_is_target"], predicted)
    else:
        outcome = "unparseable" if response_text else ""

    return {
        "trial_id": trial_id,
        "model_tag": response.get("model_tag"),
        "n_objects": cfg["n_objects"],
        "rotation_deg": cfg["rotation_deg"],
        "clockwise": cfg["clockwise"],
        "probe_on_target": cfg["probe_on_target"],
        "seed": cfg["seed"],
        "num_frames": num_frames,
        "sampling": sampling,
        "probe_is_target": ground_truth["probe_is_target"],
        "response": response_text,
        "predicted": predicted,
        "outcome": outcome,
        "prompt_tokens": response.get("prompt_tokens"),
        "output_tokens": response.get("output_tokens"),
        "generation_time_s": response.get("generation_time_s"),
        "error": response.get("error"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trials-root", type=str,
        default="data/debug_circular/gemma_frame_sweep/trials",
    )
    parser.add_argument(
        "--out-root", type=str, default=None,
        help="Defaults to results/debug_circular/gemma_frame_sweep for "
             "--response-prefix nframes, or gemma_frame_sweep_<prefix> otherwise.",
    )
    parser.add_argument(
        "--response-prefix", type=str, default="nframes", choices=["nframes", "manualframes"],
        help="Which cluster batch script's output to score -- see module docstring.",
    )
    parser.add_argument("--num-frames", type=int, nargs="+", default=[4, 8, 16, 24, 32, 50, 100])
    parser.add_argument("--sampling", type=str, nargs="+", default=["greedy", "recommended"])
    args = parser.parse_args()

    out_root = args.out_root or (
        "results/debug_circular/gemma_frame_sweep" if args.response_prefix == "nframes"
        else f"results/debug_circular/gemma_frame_sweep_{args.response_prefix}"
    )
    os.makedirs(out_root, exist_ok=True)

    if not os.path.isdir(args.trials_root):
        raise SystemExit(f"trials-root {args.trials_root} does not exist")

    trial_ids = sorted(
        d for d in os.listdir(args.trials_root)
        if os.path.isdir(os.path.join(args.trials_root, d))
    )

    rows = []
    for num_frames in args.num_frames:
        for sampling in args.sampling:
            response_file = f"cluster_response_{args.response_prefix}{num_frames}_{sampling}.json"
            for trial_id in trial_ids:
                row = load_trial_row(
                    os.path.join(args.trials_root, trial_id), trial_id, num_frames,
                    sampling, response_file,
                )
                if row is not None:
                    rows.append(row)

    n_expected = len(trial_ids) * len(args.num_frames) * len(args.sampling)
    n_found = len(rows)
    print(f"{n_found}/{n_expected} (trial, num_frames, sampling) triples scored")

    results_csv = os.path.join(out_root, "results.csv")
    with open(results_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {results_csv}")

    config_path = os.path.join(out_root, "config.json")
    with open(config_path, "w") as f:
        json.dump({
            "arm": "debug_circular",
            "run_name": "gemma_frame_sweep",
            "model": "gemma-4-31b-it",
            "trials_root": args.trials_root,
            "num_frames_swept": args.num_frames,
            "sampling_swept": args.sampling,
            "n_triples_found": n_found,
            "n_triples_expected": n_expected,
        }, f, indent=2)
    print(f"Wrote {config_path}")

    scorable = [r for r in rows if r["predicted"] is not None]
    print(f"\n--- d' by (num_frames, sampling) (n={len(scorable)} scorable triples) ---")
    by_condition = {}
    for row in scorable:
        by_condition.setdefault((row["num_frames"], row["sampling"]), []).append(row)
    for num_frames, sampling in sorted(by_condition):
        stats = compute_d_prime(by_condition[(num_frames, sampling)])
        print(
            f"num_frames={num_frames}, sampling={sampling}: d'={stats['d_prime']:.2f} "
            f"(hits={stats['hits']}, misses={stats['misses']}, "
            f"false_alarms={stats['false_alarms']}, "
            f"correct_rejections={stats['correct_rejections']}, "
            f"unparseable={stats['n_unparseable']})"
        )


if __name__ == "__main__":
    main()
