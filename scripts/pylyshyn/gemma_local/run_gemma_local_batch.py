"""
Batch video-understanding job for the locally-hosted google/gemma-4-31b-it
model on the pylyshyn n_objects={3,4,5} sweep -- see
`claude/2026_09/2026_09_10/TODO.md`. Directly mirrors
`scripts/debug_circular/run_gemma_cluster_batch.py`'s pattern, simplified to
a single fixed condition (no frame-count/sampling-mode sweep this time):
`num_frames=32` (matches OpenRouter's confirmed fixed internal cap for this
model, see the `reference_gemma_fixed_frame_budget` memory) and HuggingFace's
documented "recommended" sampling defaults (`do_sample=True, temperature=1.0,
top_p=0.95, top_k=64`).

Runs ON THE CLUSTER inside $HOME/local-llm/.venv -- self-contained, no
finst_video_model import, since that venv only has transformers/torch/etc.
installed, not this project's package. Uses the same CONFIRMED-WORKING
pattern as the debug_circular script: transformers.AutoProcessor/
AutoModelForMultimodalLM (NOT vLLM), passing the raw frame array plus
do_sample_frames=True/num_frames=32 to processor.apply_chat_template().
`build_question()` below is a standalone copy of
`scripts/pylyshyn/run_pylyshyn_trial.py`'s `build_question(cfg)` (n_cued=1
branch only, since every trial here uses n_cued=1) -- can't import it since
this script has to run without finst_video_model installed.

Processes one trial at a time via model.generate() -- no confirmed
multi-video batching pattern for this processor/model (same rationale as
the debug_circular script).

Data layout expected: <trials-root>/<trial_id>/{video.npy,ground_truth.json},
written by scripts/pylyshyn/gemma_local/generate_gemma_local_stimuli.py and rsynced up.
ground_truth.json's "config" key must have n_objects/n_cued/fps (written by
finst_video_model.pylyshyn.stimulus_gen.generate_stimulus).

    # push (local -> cluster), before submitting the job
    rsync -av --exclude='*.mp4' data/pylyshyn/gemma_local_nobjects_sweep/trials/ \\
        scotty:/mnt/cup/people/km3199/finst-video-model/data/pylyshyn/gemma_local_nobjects_sweep/trials/
    rsync -av scripts/pylyshyn/gemma_local/run_gemma_local_batch.py \\
        scotty:/mnt/cup/people/km3199/finst-video-model/scripts/pylyshyn/
    rsync -av slurm/run_gemma_pylyshyn_local.sh \\
        scotty:/mnt/cup/people/km3199/finst-video-model/slurm/

    # pull (cluster -> local), after the job finishes
    rsync -av --include='*/' --include='cluster_response.json' --exclude='*' \\
        scotty:/mnt/cup/people/km3199/finst-video-model/data/pylyshyn/gemma_local_nobjects_sweep/trials/ \\
        data/pylyshyn/gemma_local_nobjects_sweep/trials/

Writes one <trials-root>/<trial_id>/cluster_response.json per trial -- raw
model output only (parsed response text, token counts, timing), no
True/False parsing or scoring. That happens locally in
scripts/pylyshyn/gemma_local/aggregate_gemma_local_nobjects_sweep.py, where
finst_video_model.scoring is importable.

Usage (on a compute node, via slurm/run_gemma_pylyshyn_local.sh):
    python3 run_gemma_local_batch.py \\
        --trials-root /mnt/cup/people/km3199/finst-video-model/data/pylyshyn/gemma_local_nobjects_sweep/trials
"""

import argparse
import json
import os
import time
import traceback

import numpy as np
from transformers import AutoProcessor, AutoModelForMultimodalLM

MODEL_PATH = "/scratch/km3199/models/gemma-4-31B-it"
NUM_FRAMES = 32
SAMPLING = {"do_sample": True, "temperature": 1.0, "top_p": 0.95, "top_k": 64}
RESPONSE_FILE = "cluster_response.json"


def build_question(n_objects: int) -> str:
    return (
        f"You will watch a video of {n_objects} white crosses (+) on a "
        f"black background. At the start of the video, all crosses are "
        f"stationary, and 1 of them -- the cued cross -- is colored red, "
        f"while the rest stay white. After a moment it turns white too, "
        f"and every cross begins moving continuously, independently, and "
        f"unpredictably -- there is no more color difference between "
        f"crosses once they start moving, so you can only keep track of "
        f"which cross is which by following its motion. At some point "
        f"while the crosses are moving, exactly one of them briefly turns "
        f"red for about two seconds, then turns back to white, and all "
        f"crosses keep moving until the video ends.\n\n"
        f"Question: was the cross that briefly turned red the cross that "
        f"was red at the very start of the video?\n\n"
        f"Answer with only the single word True or False. Do not include "
        f"any other text in your answer."
    )


def discover_trials(trials_root: str) -> list[str]:
    return sorted(
        d for d in os.listdir(trials_root)
        if os.path.isfile(os.path.join(trials_root, d, "video.npy"))
        and os.path.isfile(os.path.join(trials_root, d, "ground_truth.json"))
    )


def already_done(trials_root: str, trial_id: str) -> bool:
    return os.path.isfile(os.path.join(trials_root, trial_id, RESPONSE_FILE))


def load_trial(trials_root: str, trial_id: str):
    trial_dir = os.path.join(trials_root, trial_id)
    frames = np.load(os.path.join(trial_dir, "video.npy"))
    with open(os.path.join(trial_dir, "ground_truth.json")) as f:
        ground_truth = json.load(f)
    return frames, ground_truth


def prepare_input(processor, frames: np.ndarray, prompt: str, native_fps: float, num_frames: int):
    assert frames.dtype == np.uint8, f"Expected uint8, got {frames.dtype}"
    assert frames.ndim == 4 and frames.shape[-1] == 3, (
        f"Expected (num_frames, H, W, 3), got {frames.shape}"
    )

    total_num_frames = frames.shape[0]
    effective_num_frames = min(num_frames, total_num_frames)
    if effective_num_frames != num_frames:
        print(
            f"  [warn] requested num_frames={num_frames} > total_num_frames="
            f"{total_num_frames}, clamping to {effective_num_frames}"
        )

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
    return inputs, effective_num_frames


def write_result(trial_dir: str, result: dict):
    tmp_path = os.path.join(trial_dir, f".{RESPONSE_FILE}.tmp")
    final_path = os.path.join(trial_dir, RESPONSE_FILE)
    with open(tmp_path, "w") as f:
        json.dump(result, f, indent=2)
    os.replace(tmp_path, final_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials-root", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    model_tag = "gemma-4-31b-it-local"

    trial_ids = discover_trials(args.trials_root)
    print(f"Discovered {len(trial_ids)} trials under {args.trials_root}")

    pending = [tid for tid in trial_ids if not already_done(args.trials_root, tid)]
    print(f"{len(trial_ids) - len(pending)}/{len(trial_ids)} trials already done (skipping), "
          f"{len(pending)} pending")
    if not pending:
        print("Nothing to do.")
        return

    print("Loading processor and model...")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_PATH,
        dtype="auto",
        device_map="auto",
    )
    print(f"Model loaded in {time.time() - t0:.1f}s")

    for i, trial_id in enumerate(pending):
        trial_dir = os.path.join(args.trials_root, trial_id)
        print(f"\n[{i + 1}/{len(pending)}] trial={trial_id}")

        try:
            frames, ground_truth = load_trial(args.trials_root, trial_id)
            cfg = ground_truth["config"]
            prompt = build_question(cfg["n_objects"])

            inputs, effective_num_frames = prepare_input(
                processor, frames, prompt, native_fps=cfg["fps"], num_frames=NUM_FRAMES,
            )
            inputs = inputs.to(model.device)
            input_len = inputs["input_ids"].shape[-1]

            t0 = time.time()
            outputs = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, **SAMPLING,
            )
            generation_time_s = time.time() - t0

            output_ids = outputs[0][input_len:]
            raw_response = processor.decode(output_ids, skip_special_tokens=False)
            prefix = processor.decode(inputs["input_ids"][0], skip_special_tokens=False)
            parsed_response = processor.parse_response(raw_response, prefix=prefix)

            result = {
                "trial_id": trial_id,
                "model_tag": model_tag,
                "num_frames_requested": NUM_FRAMES,
                "num_frames_effective": effective_num_frames,
                "prompt_tokens": input_len,
                "output_tokens": len(output_ids),
                "response_text": parsed_response,
                "raw_response_text": raw_response,
                "generation_time_s": generation_time_s,
                "error": None,
            }
            write_result(trial_dir, result)
            print(f"  {result['output_tokens']} tokens, {generation_time_s:.1f}s "
                  f"-> {parsed_response!r}")

        except Exception:
            # Deliberately don't write cluster_response.json here -- leaving
            # this trial "pending" means a resubmitted job retries it
            # automatically, rather than needing someone to find and delete
            # a stray error file first.
            print(f"  FAILED, leaving pending for retry:\n{traceback.format_exc()}")

    print("\nDone.")


if __name__ == "__main__":
    main()
