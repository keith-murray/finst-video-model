"""
Profiles how long the locally-hosted google/gemma-4-31b-it model actually
takes per (trial, num_frames) debug_circular call, so
slurm/run_gemma_frame_sweep.sh's --time can be set from a real measurement
instead of a guess (see claude/2026_09/2026_09_09/TODO.md and the qwen38
pipeline's equivalent profile_cluster_timing.py, which established the
"measure before you guess" convention this mirrors).

Runs ON THE CLUSTER inside $HOME/local-llm/.venv -- self-contained, mirrors
run_gemma_cluster_batch.py's build_question/prepare_input logic (kept
duplicated rather than imported, since both scripts must run standalone in
that venv, which doesn't have this project's package installed).

Unlike qwen3.8-27b's vLLM pipeline, this model is served directly via HF
transformers with no confirmed multi-video batching -- so this profiles
*serial* single-call timing only, one (trial, num_frames) pair at a time. The
real job (run_gemma_cluster_batch.py) also runs serially, so this measurement
matches reality (no batched-vs-serial discrepancy to worry about, unlike the
qwen profiling script's note about vLLM's continuous batching).

Usage (on a compute node, via a small sbatch wrapper or interactively):
    python3 profile_gemma_cluster_timing.py \\
        --trials-root /mnt/cup/people/km3199/finst-video-model/data/debug_circular/gemma_frame_sweep/trials \\
        --num-frames 4 8 16 24 32 50 100 \\
        --n-trials 2 \\
        --full-sweep-trials 40
"""

import argparse
import json
import os
import time

import numpy as np
from transformers import AutoProcessor, AutoModelForMultimodalLM

MODEL_PATH = "/scratch/km3199/models/gemma-4-31B-it"


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


def prepare_input(processor, frames: np.ndarray, prompt: str, native_fps: float, num_frames: int):
    total_num_frames = frames.shape[0]
    effective_num_frames = min(num_frames, total_num_frames)
    video_metadata = {
        "total_num_frames": total_num_frames,
        "fps": native_fps,
        "duration": total_num_frames / native_fps,
        "frames_indices": list(range(total_num_frames)),
    }
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": frames},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=False,
        do_sample_frames=True,
        video_metadata=video_metadata,
        num_frames=effective_num_frames,
    )
    return inputs


def load_sample_trials(trials_root: str, limit: int):
    trial_ids = sorted(
        d for d in os.listdir(trials_root)
        if os.path.isfile(os.path.join(trials_root, d, "video.npy"))
        and os.path.isfile(os.path.join(trials_root, d, "ground_truth.json"))
    )[:limit]
    trials = []
    for tid in trial_ids:
        frames = np.load(os.path.join(trials_root, tid, "video.npy"))
        with open(os.path.join(trials_root, tid, "ground_truth.json")) as f:
            gt = json.load(f)
        trials.append((tid, frames, gt))
    return trials


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trials-root", type=str,
        default="/mnt/cup/people/km3199/finst-video-model/data/debug_circular/gemma_frame_sweep/trials",
    )
    parser.add_argument("--num-frames", type=int, nargs="+", default=[4, 8, 16, 24, 32, 50, 100])
    parser.add_argument("--n-trials", type=int, default=2, help="How many sample trials to time per num_frames value.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--full-sweep-trials", type=int, default=40,
        help="Total trial count (per num_frames value) to extrapolate the full job's time for.",
    )
    args = parser.parse_args()

    trials = load_sample_trials(args.trials_root, args.n_trials)
    print(f"Loaded {len(trials)} sample trials from {args.trials_root}")
    if not trials:
        raise SystemExit(f"No trials found under {args.trials_root} -- rsync stimuli up first")

    print("Loading processor and model...")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_PATH, dtype="auto", device_map="auto",
    )
    model_load_s = time.time() - t0
    print(f"Model loaded in {model_load_s:.1f}s")

    per_trial_by_num_frames = {}
    for num_frames in args.num_frames:
        timings = []
        for tid, frames, gt in trials:
            inputs = prepare_input(
                processor, frames, build_question(gt["config"]["n_objects"]),
                gt["config"]["fps"], num_frames,
            )
            inputs = inputs.to(model.device)
            input_len = inputs["input_ids"].shape[-1]

            t0 = time.time()
            outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
            elapsed = time.time() - t0
            timings.append(elapsed)

            output_ids = outputs[0][input_len:]
            output_text = processor.decode(output_ids, skip_special_tokens=True)
            print(f"  [num_frames={num_frames}, {tid}] prompt_tokens={input_len} "
                  f"output_tokens={len(output_ids)} time={elapsed:.1f}s -> {output_text!r}")

        mean_s = sum(timings) / len(timings)
        per_trial_by_num_frames[num_frames] = mean_s
        print(f"num_frames={num_frames}: mean {mean_s:.1f}s/trial over {len(timings)} trials")

    print("\n=== Extrapolation for the full sweep ===")
    total_generation_s = sum(
        per_trial * args.full_sweep_trials for per_trial in per_trial_by_num_frames.values()
    )
    est_total_s = model_load_s + total_generation_s
    print(f"Per-num_frames mean times: {per_trial_by_num_frames}")
    print(
        f"{args.full_sweep_trials} trials x {len(args.num_frames)} num_frames values "
        f"({args.full_sweep_trials * len(args.num_frames)} total calls) = "
        f"{total_generation_s / 60:.1f} min generation + {model_load_s / 60:.1f} min load "
        f"= {est_total_s / 60:.1f} min total "
        f"(recommend --time >= {est_total_s * 1.5 / 60:.0f} min with margin)"
    )


if __name__ == "__main__":
    main()
