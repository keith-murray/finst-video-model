"""
Batch video-understanding job for the locally-hosted Qwen3.8-27B model,
adapted for the debug_circular stimulus (see
claude/2026_08/2026_08_25/TODO.md and
claude/skills/cluster/qwen38_cluster_handoff.md).

Runs ON THE CLUSTER inside $HOME/qwen38/.venv -- self-contained, no
finst_video_model import, since that venv only has vllm+huggingface_hub
installed, not this project's package. Uses the CONFIRMED-WORKING pipeline
from the handoff doc: raw .npy frame arrays passed directly via
multi_modal_data, bypassing vLLM's video_url connector entirely (see the
handoff doc's Known Bugs #5 and #6).

Data layout expected: <trials-root>/<trial_id>/{video.npy,ground_truth.json},
written by scripts/debug_circular/generate_samples.py and rsynced up.
ground_truth.json's "config" key must have n_objects/fps (written by
finst_video_model.debug_circular.stimulus_gen.generate_stimulus).

No dedicated transfer tooling exists in this repo -- run these by hand:

    # push (local -> cluster), before submitting the job
    rsync -av --exclude='*.mp4' data/debug_circular/<run_name>/trials/ \\
        scotty:/mnt/cup/people/km3199/finst-video-model/data/debug_circular/<run_name>/trials/
    rsync -av scripts/debug_circular/ \\
        scotty:/mnt/cup/people/km3199/finst-video-model/scripts/debug_circular/
    rsync -av slurm/ \\
        scotty:/mnt/cup/people/km3199/finst-video-model/slurm/

    # pull (cluster -> local), after the job finishes
    rsync -av --include='*/' --include='cluster_response.json' --exclude='*' \\
        scotty:/mnt/cup/people/km3199/finst-video-model/data/debug_circular/<run_name>/trials/ \\
        data/debug_circular/<run_name>/trials/

Writes one <trials-root>/<trial_id>/cluster_response.json per trial -- raw
model output only (response text, token counts, timing), no True/False
parsing or scoring. That happens locally in
scripts/debug_circular/aggregate_cluster_results.py, where
finst_video_model.scoring is importable.

Usage (on a compute node, via slurm/run_debug_circular_batch.sh):
    python3 run_cluster_batch.py --trials-root /mnt/cup/people/km3199/finst-video-model/data/debug_circular/<run_name>/trials
"""

import argparse
import json
import os
import time
import traceback

import numpy as np
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

MODEL_PATH = "/mnt/cup/people/km3199/models/qwen3.8-27b"
ALLOWED_LOCAL_MEDIA_PATH = "/mnt/cup/people/km3199"


def response_filename(reasoning_effort: str | None) -> str:
    return "cluster_response.json" if reasoning_effort is None else f"cluster_response_{reasoning_effort}.json"


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


def discover_trials(trials_root: str) -> list[str]:
    return sorted(
        d for d in os.listdir(trials_root)
        if os.path.isfile(os.path.join(trials_root, d, "video.npy"))
        and os.path.isfile(os.path.join(trials_root, d, "ground_truth.json"))
    )


def already_done(trials_root: str, trial_id: str, response_file: str) -> bool:
    return os.path.isfile(os.path.join(trials_root, trial_id, response_file))


def load_trial(trials_root: str, trial_id: str):
    trial_dir = os.path.join(trials_root, trial_id)
    frames = np.load(os.path.join(trial_dir, "video.npy"))
    with open(os.path.join(trial_dir, "ground_truth.json")) as f:
        ground_truth = json.load(f)
    return frames, ground_truth


def prepare_input(
    processor, frames: np.ndarray, prompt: str, native_fps: float,
    reasoning_effort: str | None = None, preserve_thinking: bool = False,
):
    assert frames.dtype == np.uint8, f"Expected uint8, got {frames.dtype}"
    assert frames.ndim == 4 and frames.shape[-1] == 3, (
        f"Expected (num_frames, H, W, 3), got {frames.shape}"
    )

    video_metadata = {
        "total_num_frames": frames.shape[0],
        "fps": native_fps,
        "duration": frames.shape[0] / native_fps,
        "frames_indices": list(range(frames.shape[0])),
        "do_sample_frames": False,
    }

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    # enable_thinking (and, when reasoning is on, preserve_thinking/
    # reasoning_effort) passed at render time too (belt-and-suspenders
    # alongside chat_template_kwargs below): `prompt` is already a rendered
    # string by the time llm.generate() sees it, so it's unconfirmed
    # whether a top-level chat_template_kwargs on the request dict has any
    # effect through that path -- see claude/2026_08/2026_08_25/TODO.md and
    # claude/2026_08/2026_08_26/TODO.md.
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
    assert width % 32 == 0 and height % 32 == 0, (
        f"width/height must be exact multiples of 32, got {width}x{height}"
    )
    tokens_per_temporal_step = (width // 32) * (height // 32)
    video_tokens = (num_frames // 2) * tokens_per_temporal_step
    prompt_budget = 300
    margin = 2400
    return video_tokens + prompt_budget + max_tokens + margin


def write_result(trial_dir: str, response_file: str, result: dict):
    tmp_path = os.path.join(trial_dir, f".{response_file}.tmp")
    final_path = os.path.join(trial_dir, response_file)
    with open(tmp_path, "w") as f:
        json.dump(result, f, indent=2)
    os.replace(tmp_path, final_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials-root", type=str, required=True)
    parser.add_argument("--chunk-size", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument(
        "--reasoning-effort", type=str, default=None, choices=["low", "medium", "xhigh"],
        help="Enables thinking mode at this effort level. Omit for the nothink baseline.",
    )
    parser.add_argument(
        "--preserve-thinking", action="store_true",
        help="Keep reasoning trace in response_text (only meaningful with --reasoning-effort). "
             "Defaults off to keep cluster_response.json small for the full batch sweep.",
    )
    args = parser.parse_args()

    model_tag = f"qwen3.8-27b-{args.reasoning_effort or 'nothink'}"
    response_file = response_filename(args.reasoning_effort)

    trial_ids = discover_trials(args.trials_root)
    print(f"Discovered {len(trial_ids)} trials under {args.trials_root}")

    pending = [
        tid for tid in trial_ids
        if not already_done(args.trials_root, tid, response_file)
    ]
    print(f"{len(trial_ids) - len(pending)} already have {response_file} "
          f"(skipping), {len(pending)} pending")
    if not pending:
        print("Nothing to do.")
        return

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    # Peek at the first pending trial to size max_model_len -- every
    # trial's shape is asserted to match this below.
    first_frames, _ = load_trial(args.trials_root, pending[0])
    num_frames, height, width = first_frames.shape[0], first_frames.shape[1], first_frames.shape[2]
    max_model_len = compute_max_model_len(width, height, num_frames, args.max_tokens)
    print(f"Video shape {first_frames.shape} -> max_model_len={max_model_len}")

    print(f"Loading model (fixed cost paid once for all {len(pending)} pending trials)...")
    t0 = time.time()
    llm = LLM(
        model=MODEL_PATH,
        dtype="bfloat16",
        tensor_parallel_size=2,
        gpu_memory_utilization=0.90,
        max_model_len=max_model_len,
        allowed_local_media_path=ALLOWED_LOCAL_MEDIA_PATH,
    )
    print(f"Model loaded in {time.time() - t0:.1f}s")

    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_tokens)

    chunks = [pending[i:i + args.chunk_size] for i in range(0, len(pending), args.chunk_size)]
    for chunk_index, chunk_ids in enumerate(chunks):
        print(f"\n=== Chunk {chunk_index + 1}/{len(chunks)} ({len(chunk_ids)} trials) ===")

        chunk_frames, chunk_gts = [], []
        for tid in chunk_ids:
            frames, ground_truth = load_trial(args.trials_root, tid)
            assert frames.shape == (num_frames, height, width, 3), (
                f"Trial {tid} has shape {frames.shape}, expected "
                f"{(num_frames, height, width, 3)} -- likely a corrupted or "
                f"mismatched-config trial, not a pipeline bug"
            )
            chunk_frames.append(frames)
            chunk_gts.append(ground_truth)

        inputs = [
            prepare_input(
                processor, frames, build_question(gt["config"]["n_objects"]),
                native_fps=gt["config"]["fps"],
                reasoning_effort=args.reasoning_effort,
                preserve_thinking=args.preserve_thinking,
            )
            for frames, gt in zip(chunk_frames, chunk_gts)
        ]

        try:
            t0 = time.time()
            outputs = llm.generate(inputs, sampling_params=sampling_params)
            elapsed = time.time() - t0
            per_trial = elapsed / len(chunk_ids)
            print(f"Chunk generated in {elapsed:.1f}s ({per_trial:.1f}s/trial)")

            for tid, output in zip(chunk_ids, outputs):
                result = {
                    "trial_id": tid,
                    "model_tag": model_tag,
                    "reasoning_effort": args.reasoning_effort,
                    "prompt_tokens": len(output.prompt_token_ids),
                    "output_tokens": len(output.outputs[0].token_ids),
                    "response_text": output.outputs[0].text,
                    "generation_time_s": per_trial,
                    "chunk_index": chunk_index,
                    "error": None,
                }
                write_result(os.path.join(args.trials_root, tid), response_file, result)
                print(f"  [{tid}] {result['output_tokens']} tokens -> {result['response_text']!r}")

        except Exception:
            # Deliberately don't write any cluster_response.json here --
            # leaving these trials "pending" means a resubmitted job
            # retries them automatically, rather than needing someone to
            # find and delete stray error files first.
            print(f"Chunk {chunk_index} failed, leaving its trials pending for retry:\n"
                  f"{traceback.format_exc()}")

    print("\nDone.")


if __name__ == "__main__":
    main()
