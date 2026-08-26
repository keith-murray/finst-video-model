"""
Profiles how long the locally-hosted Qwen3.8-27B model actually takes per
debug_circular trial, with thinking disabled, so
slurm/run_debug_circular_batch.sh's --time and run_cluster_batch.py's
--chunk-size can be set from a real measurement instead of a guess (see
claude/2026_08/2026_08_25/TODO.md's "A big unknown is how much time we
should ask for").

Runs ON THE CLUSTER inside $HOME/qwen38/.venv -- self-contained, mirrors
run_cluster_batch.py's build_question/prepare_input logic (kept duplicated
rather than imported, since both scripts must run standalone in that venv,
which doesn't have this project's package installed).

Measures *batched* throughput -- one llm.generate() call per candidate
--chunk-sizes value, not serial one-at-a-time timing. vLLM's continuous
batching means per-trial cost drops inside a batch, and the real job
(run_cluster_batch.py) always calls generate() on a chunk, never a single
trial, so a serial measurement here would give a number that doesn't match
reality.

Usage (on a compute node, via slurm/profile_debug_circular.sh):
    python3 profile_cluster_timing.py \\
        --trials-root /mnt/cup/people/km3199/finst-video-model/data/debug_circular/debug_circular_samples
"""

import argparse
import json
import os
import time

import numpy as np
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

MODEL_PATH = "/mnt/cup/people/km3199/models/qwen3.8-27b"
ALLOWED_LOCAL_MEDIA_PATH = "/mnt/cup/people/km3199"


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


def prepare_input(
    processor, frames: np.ndarray, prompt: str, native_fps: float,
    reasoning_effort: str | None = None, preserve_thinking: bool = False,
):
    video_metadata = {
        "total_num_frames": frames.shape[0],
        "fps": native_fps,
        "duration": frames.shape[0] / native_fps,
        "frames_indices": list(range(frames.shape[0])),
        "do_sample_frames": False,
    }
    messages = [
        {"role": "user", "content": [{"type": "video"}, {"type": "text", "text": prompt}]}
    ]
    if reasoning_effort is None:
        template_kwargs = {"enable_thinking": False}
    else:
        template_kwargs = {
            "enable_thinking": True,
            "preserve_thinking": preserve_thinking,
            "reasoning_effort": reasoning_effort,
        }
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **template_kwargs,
    )
    return {
        "prompt": text,
        "multi_modal_data": {"video": (frames, video_metadata)},
        "chat_template_kwargs": template_kwargs,
        "mm_processor_kwargs": {"do_sample_frames": False},
    }


def compute_max_model_len(width: int, height: int, num_frames: int, max_tokens: int) -> int:
    tokens_per_temporal_step = (width // 32) * (height // 32)
    video_tokens = (num_frames // 2) * tokens_per_temporal_step
    return video_tokens + 300 + max_tokens + 2400


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
        default="/mnt/cup/people/km3199/finst-video-model/data/debug_circular/debug_circular_samples",
    )
    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=[5, 10, 16])
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument(
        "--full-sweep-size", type=int, default=200,
        help="Total trial count to extrapolate the full job's time for",
    )
    parser.add_argument(
        "--reasoning-effort", type=str, default=None, choices=["low", "medium", "xhigh"],
        help="Enables thinking mode at this effort level. Omit for the nothink baseline. "
             "Use a much larger --max-tokens (e.g. 8192) when set.",
    )
    parser.add_argument(
        "--preserve-thinking", action="store_true",
        help="Keep the reasoning trace in the printed response text, to eyeball actual "
             "reasoning quality (only meaningful with --reasoning-effort).",
    )
    args = parser.parse_args()

    trials = load_sample_trials(args.trials_root, max(args.chunk_sizes))
    print(f"Loaded {len(trials)} sample trials from {args.trials_root}")
    if not trials:
        raise SystemExit(f"No trials found under {args.trials_root} -- rsync stimuli up first")

    chunk_sizes = [c for c in args.chunk_sizes if c <= len(trials)]
    if not chunk_sizes:
        raise SystemExit(
            f"Only {len(trials)} sample trials available, can't test any of "
            f"--chunk-sizes {args.chunk_sizes}"
        )

    first_frames = trials[0][1]
    num_frames, height, width = first_frames.shape[0], first_frames.shape[1], first_frames.shape[2]
    max_model_len = compute_max_model_len(width, height, num_frames, args.max_tokens)
    print(f"Video shape {first_frames.shape} -> max_model_len={max_model_len}")

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    print("Loading model...")
    t0 = time.time()
    llm = LLM(
        model=MODEL_PATH,
        dtype="bfloat16",
        tensor_parallel_size=2,
        gpu_memory_utilization=0.90,
        max_model_len=max_model_len,
        allowed_local_media_path=ALLOWED_LOCAL_MEDIA_PATH,
    )
    model_load_s = time.time() - t0
    print(f"Model loaded in {model_load_s:.1f}s")

    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_tokens)

    all_inputs = [
        prepare_input(
            processor, frames, build_question(gt["config"]["n_objects"]), gt["config"]["fps"],
            reasoning_effort=args.reasoning_effort, preserve_thinking=args.preserve_thinking,
        )
        for _, frames, gt in trials
    ]

    per_trial_by_chunk_size = {}
    for chunk_size in chunk_sizes:
        chunk_inputs = all_inputs[:chunk_size]
        chunk_ids = [tid for tid, _, _ in trials[:chunk_size]]

        t0 = time.time()
        outputs = llm.generate(chunk_inputs, sampling_params=sampling_params)
        elapsed = time.time() - t0
        per_trial = elapsed / chunk_size
        per_trial_by_chunk_size[chunk_size] = per_trial

        prompt_tokens = [len(o.prompt_token_ids) for o in outputs]
        output_tokens = [len(o.outputs[0].token_ids) for o in outputs]
        has_think_tag = any("<think>" in o.outputs[0].text for o in outputs)
        thinking_expected = args.reasoning_effort is not None

        print(f"\n=== chunk_size={chunk_size} reasoning_effort={args.reasoning_effort} ===")
        print(f"  total {elapsed:.1f}s, {per_trial:.1f}s/trial")
        print(f"  prompt_tokens: {prompt_tokens} (should all match)")
        print(f"  output_tokens: {output_tokens} (mean {sum(output_tokens) / len(output_tokens):.0f})")
        if thinking_expected:
            print(
                f"  any <think> tag found: {has_think_tag} "
                f"({'reasoning visible' if has_think_tag else 'not preserved (expected if --preserve-thinking omitted)'})"
            )
        else:
            print(
                f"  any <think> tag found: {has_think_tag} "
                f"({'THINKING NOT SUPPRESSED -- see TODO' if has_think_tag else 'looks suppressed'})"
            )
        for tid, o in zip(chunk_ids, outputs):
            print(f"    [{tid}] {o.outputs[0].text!r}")

    print("\n=== Extrapolation for the full sweep ===")
    for chunk_size, per_trial in per_trial_by_chunk_size.items():
        est_generation_s = per_trial * args.full_sweep_size
        est_total_s = model_load_s + est_generation_s
        print(
            f"chunk_size={chunk_size}: {per_trial:.1f}s/trial -> "
            f"{args.full_sweep_size} trials = {est_generation_s / 60:.1f} min generation "
            f"+ {model_load_s / 60:.1f} min load = {est_total_s / 60:.1f} min total "
            f"(recommend --time >= {est_total_s * 1.5 / 60:.0f} min with margin)"
        )


if __name__ == "__main__":
    main()
