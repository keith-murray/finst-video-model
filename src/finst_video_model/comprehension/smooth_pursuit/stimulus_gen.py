"""
Renders the complete trial video for the smooth-pursuit task:
  1. Cue phase (cfg.cue_flash_s): circles stationary. K circles red
     (cued), rest gray. Matches Pylyshyn's "cueing happens before motion
     starts, not during it" -- no `step` calls happen in this phase.
  2. Tracking phase (cfg.tracking_s): all circles identical gray, moving
     with one constant heading each for the whole phase (the "smooth"
     motion that distinguishes this task from the pylyshyn arm's frequent
     redirects). A continuous minimum-separation constraint
     (`physics.step_tracking_frame`) runs every frame.
  3. Probe phase (cfg.probe_flash_s): motion freezes at its final position,
     and exactly one circle (probed_index, chosen per cfg.probe_on_target)
     is recolored cued_color; all others stay neutral gray. The model is
     asked a single True/False question -- "was this circle cued at the
     start?" -- rather than the old label phase's "name every cued letter."

Ground truth (which circles were cued, which one was probed, and whether
that probe was actually cued) is known exactly, since we simulated every
position ourselves.

Writes `<out_dir>/video.mp4` and `<out_dir>/ground_truth.json`. `out_dir` is
expected to already be trial-specific (e.g. `data/<trial_id>/`), matching
the convention used elsewhere in this project.
"""

import json
import random
from dataclasses import asdict

import cv2
import numpy as np

from finst_video_model.comprehension.smooth_pursuit.config import TrialConfig
from finst_video_model.comprehension.smooth_pursuit.physics import build_circles, step_tracking_frame


def _draw_frame(cfg: TrialConfig, circles, cued_indices, show_cue: bool = False,
                 probed_index: int | None = None) -> np.ndarray:
    img = np.full((cfg.image_height, cfg.image_width, 3), cfg.background_color, dtype=np.uint8)

    for c in circles:
        if c.index == probed_index or (show_cue and c.index in cued_indices):
            color = cfg.cued_color
        else:
            color = cfg.neutral_color

        center = (int(round(c.x)), int(round(c.y)))
        cv2.circle(img, center, cfg.circle_radius, color[::-1], thickness=-1)  # BGR
        cv2.circle(img, center, cfg.circle_radius, (0, 0, 0), thickness=2)

    return img


def _choose_probe(cfg: TrialConfig, cued_indices: set[int], rng: random.Random) -> int:
    candidates = sorted(cued_indices if cfg.probe_on_target else set(range(cfg.n_circles)) - cued_indices)
    return rng.choice(candidates)


def generate_stimulus(cfg: TrialConfig, out_dir: str) -> dict:
    cfg.validate()

    circles, cued_indices = build_circles(cfg)
    probe_rng = random.Random(cfg.seed + 2)
    probed_index = _choose_probe(cfg, cued_indices, probe_rng)

    dt = 1.0 / cfg.fps
    n_cue_frames = int(round(cfg.cue_flash_s * cfg.fps))
    n_track_frames = int(round(cfg.tracking_s * cfg.fps))
    n_probe_frames = int(round(cfg.probe_flash_s * cfg.fps))

    video_path = f"{out_dir}/video.mp4"
    # avc1 (H.264) rather than mp4v (raw MPEG-4 Part 2) -- mp4v is valid but
    # QuickTime/macOS's default player often can't decode it correctly
    # (renders as solid green), even though standards-compliant decoders
    # (e.g. the ffmpeg backend cv2.VideoCapture itself uses) read it fine.
    writer = cv2.VideoWriter(
        video_path, cv2.VideoWriter_fourcc(*"avc1"), cfg.fps,
        (cfg.image_width, cfg.image_height)
    )

    # --- Cue phase: stationary, cued subset shown red ---
    for _ in range(n_cue_frames):
        writer.write(_draw_frame(cfg, circles, cued_indices, show_cue=True))

    # --- Tracking phase: all circles gray, moving ---
    for _ in range(n_track_frames):
        writer.write(_draw_frame(cfg, circles, cued_indices))
        step_tracking_frame(cfg, circles, dt)

    # --- Probe phase: freeze, highlight probed_index, hold ---
    probe_frame = _draw_frame(cfg, circles, cued_indices, probed_index=probed_index)
    for _ in range(n_probe_frames):
        writer.write(probe_frame)

    writer.release()

    circles_meta = [
        {
            "index": c.index,
            "final_x": round(c.x, 1),
            "final_y": round(c.y, 1),
            "cued": c.index in cued_indices,
        }
        for c in circles
    ]

    ground_truth = {
        "trial_id": cfg.trial_id,
        "config": asdict(cfg),
        "video_path": video_path,
        "circles": circles_meta,
        "cued_indices": sorted(cued_indices),
        "probed_index": probed_index,
        "probe_is_target": probed_index in cued_indices,
    }

    gt_path = f"{out_dir}/ground_truth.json"
    with open(gt_path, "w") as f:
        json.dump(ground_truth, f, indent=2)

    return ground_truth
