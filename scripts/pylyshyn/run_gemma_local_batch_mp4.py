"""
Batch video-understanding job for the locally-hosted google/gemma-4-31b-it
model on the pylyshyn n_objects={3,4,5} sweep -- see
`claude/2026_09/2026_09_14/TODO.md`. Sibling of
`scripts/pylyshyn/run_gemma_local_batch.py`, but feeds the model the native
`video.mp4` directly instead of a pre-decoded `video.npy` array + hand-built
`video_metadata` dict: `{"type": "video", "path": video_path}` lets
transformers' own video pipeline (decoded via torchcodec, per the user's
cluster-side torchcodec/ffmpeg setup -- see
`claude/skills/cluster/ffmpeg/test_torchcodec_with_module.sh`) extract frame
sampling, fps, and duration itself. No `video_metadata`, no
`do_sample_frames`, no manual frame-count clamping.

Runs ON THE CLUSTER inside $HOME/local-llm/.venv -- self-contained, no
finst_video_model import, since that venv only has transformers/torch/etc.
installed, not this project's package. `build_question()` below is a
standalone copy of `scripts/pylyshyn/run_pylyshyn_trial.py`'s
`build_question(cfg)` (n_cued=1 branch only, since every trial here uses
n_cued=1) -- can't import it since this script has to run without
finst_video_model installed.

Processes one trial at a time via model.generate() -- no confirmed
multi-video batching pattern for this processor/model.

Data layout expected: <trials-root>/<trial_id>/{video.mp4,ground_truth.json}.
`video.mp4` is already written by
scripts/pylyshyn/generate_gemma_local_stimuli.py (alongside video.npy, used
by the sibling npy-based script) and was already rsynced to the cluster
ahead of this task -- no new stimulus generation or push needed for trial
data. ground_truth.json's "config" key must have n_objects/n_cued/fps
(written by finst_video_model.pylyshyn.stimulus_gen.generate_stimulus).

    # push (local -> cluster), before submitting the job
    rsync -av scripts/pylyshyn/run_gemma_local_batch_mp4.py \\
        scotty:/mnt/cup/people/km3199/finst-video-model/scripts/pylyshyn/
    rsync -av slurm/run_gemma_pylyshyn_local_mp4.sh \\
        scotty:/mnt/cup/people/km3199/finst-video-model/slurm/

    # pull (cluster -> local), after the job finishes
    rsync -av --include='*/' --include='cluster_response_mp4.json' --exclude='*' \\
        scotty:/mnt/cup/people/km3199/finst-video-model/data/pylyshyn/gemma_local_nobjects_sweep/trials/ \\
        data/pylyshyn/gemma_local_nobjects_sweep/trials/

Writes one <trials-root>/<trial_id>/cluster_response_mp4.json per trial --
raw model output only (parsed response text, token counts, timing), no
True/False parsing or scoring. Deliberately a distinct filename from the
npy-based script's cluster_response.json, so the two runs' resume-skip
logic and aggregation never conflate. Scoring happens locally in
scripts/pylyshyn/aggregate_gemma_local_mp4_nobjects_sweep.py, where
finst_video_model.scoring is importable.

Usage (on a compute node, via slurm/run_gemma_pylyshyn_local_mp4.sh):
    python3 run_gemma_local_batch_mp4.py \\
        --trials-root /mnt/cup/people/km3199/finst-video-model/data/pylyshyn/gemma_local_nobjects_sweep/trials
"""

import argparse
import json
import os
import time
import traceback

from transformers import AutoProcessor, AutoModelForMultimodalLM

MODEL_PATH = "/scratch/km3199/models/gemma-4-31B-it"
NUM_FRAMES = 32
SAMPLING = {"do_sample": True, "temperature": 1.0, "top_p": 0.95, "top_k": 64}
RESPONSE_FILE = "cluster_response_mp4.json"


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
        if os.path.isfile(os.path.join(trials_root, d, "video.mp4"))
        and os.path.isfile(os.path.join(trials_root, d, "ground_truth.json"))
    )


def already_done(trials_root: str, trial_id: str) -> bool:
    return os.path.isfile(os.path.join(trials_root, trial_id, RESPONSE_FILE))


def load_trial(trials_root: str, trial_id: str):
    trial_dir = os.path.join(trials_root, trial_id)
    video_path = os.path.join(trial_dir, "video.mp4")
    with open(os.path.join(trial_dir, "ground_truth.json")) as f:
        ground_truth = json.load(f)
    return video_path, ground_truth


def prepare_input(processor, video_path: str, prompt: str, num_frames: int):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "path": video_path},
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
        num_frames=num_frames,
    )
    return inputs


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

    model_tag = "gemma-4-31b-it-local-mp4"

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
            video_path, ground_truth = load_trial(args.trials_root, trial_id)
            cfg = ground_truth["config"]
            prompt = build_question(cfg["n_objects"])

            inputs = prepare_input(processor, video_path, prompt, num_frames=NUM_FRAMES)
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
            # Deliberately don't write cluster_response_mp4.json here --
            # leaving this trial "pending" means a resubmitted job retries
            # it automatically, rather than needing someone to find and
            # delete a stray error file first.
            print(f"  FAILED, leaving pending for retry:\n{traceback.format_exc()}")

    print("\nDone.")


if __name__ == "__main__":
    main()
