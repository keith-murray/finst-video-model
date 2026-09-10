"""
Diagnostic variant of run_gemma_cluster_batch.py that BYPASSES
do_sample_frames entirely and manually selects which frames to feed
google/gemma-4-31b-it, instead of trusting processor.apply_chat_template()'s
do_sample_frames=True/num_frames=N auto-sampler.

Why this exists (see claude/2026_09/2026_09_09/SUMMARY.md and
project_status_2026_09_09 memory for the full writeup): the first two real
cluster runs via run_gemma_cluster_batch.py came back far below OpenRouter's
~93% accuracy at every num_frames, including num_frames=100 (the full,
untruncated native clip -- no subsampling possible, yet still far worse than
OpenRouter's own 32-frame-capped result on the identical stimulus). Sampling
mode (greedy vs. recommended) and prompt wording were both directly ruled out
as the cause (near-identical results / byte-identical text to
run_openrouter_diagnostic.py's build_question). That leaves the do_sample_frames
auto-sampler itself as the leading suspect -- there is a DIRECT precedent for
exactly this failure mode already documented in this repo:
qwen38_cluster_handoff.md's Known Bug #5, where vLLM's video_url connector
silently pre-decoded/truncated video to the wrong frame count while still
reporting correct-looking metadata. The fix there was to bypass automatic
sampling entirely and hand-select exact frames -- this script applies the
same fix to gemma's HF-transformers pipeline as a diagnostic, kept as a
SEPARATE script from run_gemma_cluster_batch.py (per the user's explicit
request) rather than modifying the auto-sampling version in place, so both
approaches' results can be compared head-to-head under the same trials/data.

Frame selection: N indices evenly spaced across the full clip via
np.linspace(0, total_num_frames - 1, N), rounded and de-duplicated (dedup can
only make the effective count slightly less than N, logged when it happens).
The sliced sub-array is then passed to the processor with do_sample_frames=False
(no further resampling) -- what the model receives is now exactly and only
what this script chose. Prints the selected frame indices to the job log for
every (trial, num_frames) pair, so the actual sampling can be eyeballed
directly this time instead of inferred.

Video metadata for the sliced sub-array: this script reports the REAL elapsed
duration of the original clip (not a compressed N-frames-at-native-fps
duration), with fps recomputed as N / real_duration -- i.e. "here are N frames
representative of what happened over the real T seconds", not "here is a
brand new T*N/total-second video". This mirrors how this project's duration-
stretch-trick experiments (reference_openrouter_duration_stretch_trick) treat
fps/duration as the channel that communicates real-world timing separately
from frame count. This choice is itself an assumption, not yet verified --
if bypassing do_sample_frames doesn't fix the accuracy gap, revisit whether
the OTHER metadata convention (native fps, compressed duration) fares
differently.

Runs ON THE CLUSTER inside $HOME/local-llm/.venv, same environment/model path
as run_gemma_cluster_batch.py. Data layout, rsync commands, and CLI shape are
otherwise identical to that script -- see its docstring for the full
transfer-command reference (adjust filenames per below).

Writes one <trials-root>/<trial_id>/cluster_response_manualframes{N}_{sampling}.json
per (trial, num_frames, sampling) triple -- distinct filename prefix from
run_gemma_cluster_batch.py's cluster_response_nframes{N}_{sampling}.json so
both pipelines' data can coexist under the same trials directory. Score with
scripts/debug_circular/aggregate_gemma_frame_sweep.py's --response-prefix
manualframes flag.

Usage (on a compute node):
    python3 run_gemma_cluster_batch_manual_frames.py \\
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

SAMPLING_CONFIGS = {
    "greedy": {"do_sample": False},
    "recommended": {"do_sample": True, "temperature": 1.0, "top_p": 0.95, "top_k": 64},
}


def response_filename(num_frames: int, sampling: str) -> str:
    return f"cluster_response_manualframes{num_frames}_{sampling}.json"


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


def select_frame_indices(total_num_frames: int, num_frames: int) -> np.ndarray:
    """N indices evenly spaced across [0, total_num_frames - 1], deduplicated."""
    raw = np.linspace(0, total_num_frames - 1, num_frames)
    indices = np.unique(np.round(raw).astype(int))
    return indices


def prepare_input(
    processor, frames: np.ndarray, prompt: str, native_fps: float, num_frames: int,
):
    assert frames.dtype == np.uint8, f"Expected uint8, got {frames.dtype}"
    assert frames.ndim == 4 and frames.shape[-1] == 3, (
        f"Expected (num_frames, H, W, 3), got {frames.shape}"
    )

    total_num_frames = frames.shape[0]
    real_duration_s = total_num_frames / native_fps
    requested_num_frames = min(num_frames, total_num_frames)

    indices = select_frame_indices(total_num_frames, requested_num_frames)
    effective_num_frames = len(indices)
    if effective_num_frames != requested_num_frames:
        print(
            f"  [warn] requested {requested_num_frames} frames but dedup of "
            f"rounded indices gave {effective_num_frames} unique frames: "
            f"{indices.tolist()}"
        )
    print(f"  selected frame indices: {indices.tolist()}")

    selected_frames = frames[indices]

    # Preserve real elapsed duration, recompute an effective fps for the
    # subsampled frame count -- see module docstring for why (an assumption,
    # not yet verified against the alternative "compressed duration" reading).
    effective_fps = effective_num_frames / real_duration_s

    video_metadata = {
        "total_num_frames": effective_num_frames,
        "fps": effective_fps,
        "duration": real_duration_s,
        "frames_indices": list(range(effective_num_frames)),
    }

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": selected_frames},
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
        do_sample_frames=False,
        video_metadata=video_metadata,
    )
    return inputs, effective_num_frames, indices.tolist()


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

            inputs, effective_num_frames, selected_indices = prepare_input(
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
                "selected_frame_indices": selected_indices,
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
            # Deliberately don't write any cluster_response_manualframes{N}_{sampling}.json
            # here -- leaving this triple "pending" means a resubmitted job
            # retries it automatically, rather than needing someone to find
            # and delete a stray error file first.
            print(f"  FAILED, leaving pending for retry:\n{traceback.format_exc()}")

    print("\nDone.")


if __name__ == "__main__":
    main()
