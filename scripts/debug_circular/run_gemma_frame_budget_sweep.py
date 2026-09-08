"""
Hand-constructs minimal videos with exactly N real frames (N = 1..15) and
sends each to google/gemma-4-31b-it via OpenRouter to see how
`usage.prompt_tokens` scales with real frame count -- directly probing
gemma's fixed frame-sampling budget (see claude/2026_09/2026_09_08/TODO.md),
rather than the indirect duration/fps-based check in
run_debug_circular_angle_sweep.py's sibling scripts (which varied a real
clip's declared duration/fps but never its actual frame count this finely).

Prior data (see reference_gemma_fixed_frame_budget memory) showed
`prompt_tokens` is exactly identical across 100/200/400-frame real clips --
i.e. gemma extracts a fixed-size sample regardless of how much content
exists. If that budget is, say, ~9 frames, then feeding it fewer real frames
than the budget should force `prompt_tokens` to actually track frame count
(nothing to discard yet), then flatten once frame count exceeds the budget.

Each video is just a single white cross drifting a few pixels per frame on a
black 384x384 background (reusing debug_circular.stimulus_gen's
`_draw_cross`/`write_mp4`) -- content is deliberately trivial, and every
frame is made visually distinct (not static) only so nothing downstream
(codec-level or provider-side dedup) can collapse repeated frames and
confound the frame-count signal. A single generic, task-agnostic question is
reused across every n_frames value so text-token overhead stays constant and
doesn't confound the video-token comparison; the response text is logged for
the record but not scored.

2 reps per n_frames (different drift start position) sanity-check that
prompt_tokens is deterministic given frame count, as prior data already
suggested, rather than serving as a statistical sample.

Every trial's raw video lands in data/debug_circular/frame_budget_sweep/
(gitignored); results.csv (tracked) goes to
results/debug_circular/frame_budget_sweep/. results.csv is written
incrementally and a re-run resumes automatically (already-run
(n_frames, rep) pairs are skipped).

Usage:
    uv run python scripts/debug_circular/run_gemma_frame_budget_sweep.py
    uv run python scripts/debug_circular/run_gemma_frame_budget_sweep.py --max-frames 20
"""

import argparse
import csv
import os

import numpy as np
import requests

from finst_video_model.debug_circular.stimulus_gen import _draw_cross, write_mp4
from finst_video_model.vlm_client import ask_about_video

RUN_NAME = "frame_budget_sweep"
MODEL = "google/gemma-4-31b-it"
QUESTION = "Describe what you see in this video in one sentence."
IMAGE_SIZE = 384  # matches debug_circular.config.TrialConfig's defaults
CROSS_SIZE = 14
CROSS_THICKNESS = 5
FPS = 10
N_REPS = 2

RESULT_FIELDS = [
    "n_frames", "rep", "model", "response",
    "prompt_tokens", "completion_tokens", "cost", "error",
]


def build_video(n_frames: int, rep: int, path: str):
    start_x = 60 + rep * 40  # different start position per rep
    frames = []
    for i in range(n_frames):
        img = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        x = start_x + i * 12
        y = IMAGE_SIZE // 2
        _draw_cross(img, x, y, CROSS_SIZE, CROSS_THICKNESS, color=(255, 255, 255))
        frames.append(img)
    write_mp4(np.stack(frames, axis=0), path, fps=FPS)


def run_one(data_dir: str, n_frames: int, rep: int) -> dict:
    video_path = os.path.join(data_dir, f"n{n_frames:02d}_rep{rep}.mp4")
    build_video(n_frames, rep, video_path)

    row = {
        "n_frames": n_frames, "rep": rep, "model": MODEL,
        "response": "", "prompt_tokens": "", "completion_tokens": "",
        "cost": "", "error": "",
    }
    try:
        response_text, usage = ask_about_video(
            video_path, QUESTION, MODEL,
            reasoning={"enabled": False}, return_usage=True,
        )
        row.update({
            "response": response_text,
            "prompt_tokens": (usage or {}).get("prompt_tokens"),
            "completion_tokens": (usage or {}).get("completion_tokens"),
            "cost": (usage or {}).get("cost"),
        })
    except (requests.RequestException, KeyError, RuntimeError) as e:
        row["error"] = str(e)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-frames", type=int, default=15)
    parser.add_argument("--reps", type=int, default=N_REPS)
    args = parser.parse_args()

    data_dir = os.path.join("data", "debug_circular", RUN_NAME)
    results_dir = os.path.join("results", "debug_circular", RUN_NAME)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    results_csv = os.path.join(results_dir, "results.csv")
    already_ran = set()
    file_exists = os.path.isfile(results_csv)
    if file_exists:
        with open(results_csv) as f:
            for r in csv.DictReader(f):
                already_ran.add((int(r["n_frames"]), int(r["rep"])))

    grid = [
        (n_frames, rep)
        for n_frames in range(1, args.max_frames + 1)
        for rep in range(args.reps)
        if (n_frames, rep) not in already_ran
    ]
    total = args.max_frames * args.reps
    done = len(already_ran)

    with open(results_csv, "a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=RESULT_FIELDS)
        if not file_exists:
            writer.writeheader()
            csv_file.flush()

        for n_frames, rep in grid:
            row = run_one(data_dir, n_frames, rep)
            done += 1
            print(
                f"[{done}/{total}] n_frames={n_frames} rep={rep} "
                f"prompt_tokens={row['prompt_tokens']} error={row['error']}"
            )
            writer.writerow(row)
            csv_file.flush()

    print(f"\nresults.csv up to date at {results_csv}")


if __name__ == "__main__":
    main()
