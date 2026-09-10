"""
Batch video-understanding job for the locally-hosted google/gemma-4-31b-it
model, sweeping the number of frames sampled from each debug_circular video
AND the generation sampling mode (see claude/2026_09/2026_09_09/TODO.md and
claude/skills/cluster/gemma4/test_gemma4_baseline.py/.sh).

Two sampling modes (SAMPLING_CONFIGS below): "greedy" (do_sample=False,
deterministic, matches the qwen3.8-27b cluster pipeline's
temperature=0.0/top_p=1.0 convention) and "recommended" (do_sample=True,
temperature=1.0, top_p=0.95, top_k=64 -- HuggingFace's documented defaults for
this model). Neither test_gemma4_baseline.py nor this script's first draft set
any sampling params at all, silently falling back to whatever
generation_config.json ships with the checkpoint -- both modes here are now
explicit rather than left to that unverified default. "recommended" is
stochastic: each trial gets exactly one draw, not averaged over repeated
samples, so treat its per-condition accuracy as noisier than "greedy"'s.

Runs ON THE CLUSTER inside $HOME/local-llm/.venv -- self-contained, no
finst_video_model import, since that venv only has transformers/torch/etc.
installed, not this project's package. Uses the CONFIRMED-WORKING pattern
from test_gemma4_baseline.py: transformers.AutoProcessor/
AutoModelForMultimodalLM (NOT vLLM -- this model is hosted directly via HF
transformers, unlike the qwen3.8-27b cluster pipeline), passing the raw frame
array plus do_sample_frames=True/num_frames=N to
processor.apply_chat_template() to control exactly how many frames the model
samples.

Unlike run_cluster_batch.py's vLLM-based batched llm.generate(), this script
processes one (trial, num_frames) pair at a time via model.generate() -- there
is no confirmed multi-video batching pattern for this processor/model, and
test_gemma4_baseline.py only demonstrates single-video generation. If per-call
timing (see profile_gemma_cluster_timing.py) makes the full sweep too slow,
batching could be revisited, but serial-first avoids introducing an untested
batching bug into the results.

Data layout expected: <trials-root>/<trial_id>/{video.npy,ground_truth.json},
written by scripts/debug_circular/generate_samples.py and rsynced up.
ground_truth.json's "config" key must have n_objects/fps (written by
finst_video_model.debug_circular.stimulus_gen.generate_stimulus).

No dedicated transfer tooling exists in this repo -- run these by hand. NOTE:
the /mnt/cup/... path below matches the convention established by the
qwen3.8-27b cluster pipeline (see qwen38_cluster_handoff.md) -- confirm this
is also where the gemma4 environment expects the repo to live before running,
since test_gemma4_baseline.py's own hardcoded sample path
(/usr/people/km3199/finst-video-model/...) suggests it may differ:

    # push (local -> cluster), before submitting the job
    rsync -av --exclude='*.mp4' data/debug_circular/gemma_frame_sweep/trials/ \\
        scotty:/mnt/cup/people/km3199/finst-video-model/data/debug_circular/gemma_frame_sweep/trials/
    rsync -av scripts/debug_circular/run_gemma_cluster_batch.py \\
        scotty:/mnt/cup/people/km3199/finst-video-model/scripts/debug_circular/
    rsync -av slurm/run_gemma_frame_sweep.sh \\
        scotty:/mnt/cup/people/km3199/finst-video-model/slurm/

    # pull (cluster -> local), after the job finishes
    rsync -av --include='*/' --include='cluster_response_nframes*.json' --exclude='*' \\
        scotty:/mnt/cup/people/km3199/finst-video-model/data/debug_circular/gemma_frame_sweep/trials/ \\
        data/debug_circular/gemma_frame_sweep/trials/

Writes one <trials-root>/<trial_id>/cluster_response_nframes{N}_{sampling}.json
per (trial, num_frames, sampling) triple -- raw model output only (parsed
response text, token counts, timing), no True/False parsing or scoring. That
happens locally in scripts/debug_circular/aggregate_gemma_frame_sweep.py,
where finst_video_model.scoring is importable.

Usage (on a compute node, via slurm/run_gemma_frame_sweep.sh):
    python3 run_gemma_cluster_batch.py \\
        --trials-root /mnt/cup/people/km3199/finst-video-model/data/debug_circular/gemma_frame_sweep/trials \\
        --num-frames 4 8 16 24 32 50 100 \\
        --sampling greedy recommended
"""

import argparse
import json
import os
import time
import traceback

import numpy as np
from transformers import AutoProcessor, AutoModelForMultimodalLM

MODEL_PATH = "/scratch/km3199/models/gemma-4-31B-it"

# "greedy": deterministic, matches the qwen3.8-27b cluster pipeline's
# temperature=0.0/top_p=1.0 convention. "recommended": HuggingFace's
# documented sampling defaults for this model card.
SAMPLING_CONFIGS = {
    "greedy": {"do_sample": False},
    "recommended": {"do_sample": True, "temperature": 1.0, "top_p": 0.95, "top_k": 64},
}


def response_filename(num_frames: int, sampling: str) -> str:
    return f"cluster_response_nframes{num_frames}_{sampling}.json"


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
    processor, frames: np.ndarray, prompt: str, native_fps: float, num_frames: int,
):
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


def write_result(trial_dir: str, response_file: str, result: dict):
    tmp_path = os.path.join(trial_dir, f".{response_file}.tmp")
    final_path = os.path.join(trial_dir, response_file)
    with open(tmp_path, "w") as f:
        json.dump(result, f, indent=2)
    os.replace(tmp_path, final_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials-root", type=str, required=True)
    parser.add_argument(
        "--num-frames", type=int, nargs="+", required=True,
        help="One or more frame-sampling budgets to sweep, e.g. 4 8 16 24 32 50 100.",
    )
    parser.add_argument(
        "--sampling", type=str, nargs="+", default=["greedy", "recommended"],
        choices=list(SAMPLING_CONFIGS),
        help="One or more generation sampling modes to sweep (see SAMPLING_CONFIGS).",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    model_tag = "gemma-4-31b-it"

    trial_ids = discover_trials(args.trials_root)
    print(f"Discovered {len(trial_ids)} trials under {args.trials_root}")

    triples = [
        (tid, nf, sampling)
        for tid in trial_ids
        for nf in args.num_frames
        for sampling in args.sampling
        if not already_done(args.trials_root, tid, response_filename(nf, sampling))
    ]
    n_total = len(trial_ids) * len(args.num_frames) * len(args.sampling)
    print(f"{n_total - len(triples)}/{n_total} (trial, num_frames, sampling) triples "
          f"already done (skipping), {len(triples)} pending")
    if not triples:
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

    for i, (trial_id, num_frames, sampling) in enumerate(triples):
        response_file = response_filename(num_frames, sampling)
        trial_dir = os.path.join(args.trials_root, trial_id)
        print(f"\n[{i + 1}/{len(triples)}] trial={trial_id} num_frames={num_frames} "
              f"sampling={sampling}")

        try:
            frames, ground_truth = load_trial(args.trials_root, trial_id)
            cfg = ground_truth["config"]
            prompt = build_question(cfg["n_objects"])

            inputs, effective_num_frames = prepare_input(
                processor, frames, prompt, native_fps=cfg["fps"], num_frames=num_frames,
            )
            inputs = inputs.to(model.device)
            input_len = inputs["input_ids"].shape[-1]

            t0 = time.time()
            outputs = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, **SAMPLING_CONFIGS[sampling],
            )
            generation_time_s = time.time() - t0

            output_ids = outputs[0][input_len:]
            raw_response = processor.decode(output_ids, skip_special_tokens=False)
            prefix = processor.decode(inputs["input_ids"][0], skip_special_tokens=False)
            parsed_response = processor.parse_response(raw_response, prefix=prefix)

            result = {
                "trial_id": trial_id,
                "model_tag": model_tag,
                "num_frames_requested": num_frames,
                "num_frames_effective": effective_num_frames,
                "sampling": sampling,
                "prompt_tokens": input_len,
                "output_tokens": len(output_ids),
                "response_text": parsed_response,
                "raw_response_text": raw_response,
                "generation_time_s": generation_time_s,
                "error": None,
            }
            write_result(trial_dir, response_file, result)
            print(f"  {result['output_tokens']} tokens, {generation_time_s:.1f}s "
                  f"-> {parsed_response!r}")

        except Exception:
            # Deliberately don't write any cluster_response_nframes{N}_{sampling}.json
            # here -- leaving this triple "pending" means a resubmitted job
            # retries it automatically, rather than needing someone to find
            # and delete a stray error file first.
            print(f"  FAILED, leaving pending for retry:\n{traceback.format_exc()}")

    print("\nDone.")


if __name__ == "__main__":
    main()
