"""
Renders the complete trial video for the Pylyshyn reproduction:
  1. Cue phase (cfg.cue_s): all objects stationary, drawn as crosses. The
     cued subset blinks (on/off every cfg.cue_blink_period_s); distractors
     stay steadily drawn. No motion during this phase, matching the
     original ("After 10 seconds the flashing stopped and all 10 objects
     began moving").
  2. Tracking phase (cfg.tracking_s): all objects move identically (no
     color/blink distinction -- cueing information must be tracked purely
     by motion from here on), via continuously-redirecting random-walk
     physics (see `physics.py`). At a single randomly chosen moment within
     this phase, one object -- either a cued target or a distractor,
     depending on cfg.probe_on_target -- is drawn as a solid square instead
     of a cross for cfg.probe_flash_s, then reverts to a cross and motion
     continues to the end of the phase.

Ground truth (which objects were cued, which one was probed, and whether
that probe was a target) is known exactly, since we simulated every
position ourselves.

Writes `<out_dir>/video.mp4` and `<out_dir>/ground_truth.json`, matching the
convention used by the other arms.
"""

import json
import random
from dataclasses import asdict

import cv2
import numpy as np

from finst_video_model.comprehension.pylyshyn.config import TrialConfig
from finst_video_model.comprehension.pylyshyn.physics import build_objects, step_tracking_frame


def _draw_cross(img, x, y, size, thickness, color):
    center = (int(round(x)), int(round(y)))
    half_t = thickness // 2
    cv2.rectangle(
        img, (center[0] - size, center[1] - half_t), (center[0] + size, center[1] + half_t),
        color[::-1], thickness=-1,
    )
    cv2.rectangle(
        img, (center[0] - half_t, center[1] - size), (center[0] + half_t, center[1] + size),
        color[::-1], thickness=-1,
    )


def _draw_square(img, x, y, size, color):
    center = (int(round(x)), int(round(y)))
    cv2.rectangle(
        img, (center[0] - size, center[1] - size), (center[0] + size, center[1] + size),
        color[::-1], thickness=-1,
    )


def _draw_frame(cfg: TrialConfig, objects, blink_on: bool, cued_indices,
                 probed_index: int | None = None) -> np.ndarray:
    img = np.full((cfg.image_height, cfg.image_width, 3), cfg.background_color, dtype=np.uint8)

    for obj in objects:
        if probed_index is not None and obj.index == probed_index:
            _draw_square(img, obj.x, obj.y, cfg.object_size, cfg.probe_color)
            continue
        if obj.index in cued_indices and not blink_on:
            continue  # cued object mid-blink-off: not drawn
        _draw_cross(img, obj.x, obj.y, cfg.object_size, cfg.arm_thickness, cfg.object_color)

    return img


def _choose_probe(cfg: TrialConfig, cued_indices: set[int], rng: random.Random):
    candidates = sorted(cued_indices if cfg.probe_on_target else set(range(cfg.n_objects)) - cued_indices)
    probed_index = rng.choice(candidates)
    t_probe = rng.uniform(cfg.min_probe_delay_s, cfg.tracking_s - cfg.min_post_probe_s)
    return probed_index, t_probe


def generate_stimulus(cfg: TrialConfig, out_dir: str) -> dict:
    cfg.validate()

    objects, cued_indices = build_objects(cfg)
    probe_rng = random.Random(cfg.seed + 2)
    probed_index, t_probe = _choose_probe(cfg, cued_indices, probe_rng)

    dt = 1.0 / cfg.fps
    n_cue_frames = int(round(cfg.cue_s * cfg.fps))
    n_track_frames = int(round(cfg.tracking_s * cfg.fps))
    n_probe_frames = max(1, int(round(cfg.probe_flash_s * cfg.fps)))
    probe_start_frame = int(round(t_probe * cfg.fps))

    video_path = f"{out_dir}/video.mp4"
    writer = cv2.VideoWriter(
        video_path, cv2.VideoWriter_fourcc(*"avc1"), cfg.fps,
        (cfg.image_width, cfg.image_height)
    )

    # --- Cue phase: stationary, cued subset blinks ---
    for frame_i in range(n_cue_frames):
        t = frame_i * dt
        blink_on = int(t / cfg.cue_blink_period_s) % 2 == 0
        writer.write(_draw_frame(cfg, objects, blink_on, cued_indices))

    # --- Tracking phase: all objects move; one probe flash mid-phase ---
    for frame_i in range(n_track_frames):
        t = frame_i * dt
        is_probing = probe_start_frame <= frame_i < probe_start_frame + n_probe_frames
        writer.write(_draw_frame(
            cfg, objects, blink_on=True, cued_indices=cued_indices,
            probed_index=probed_index if is_probing else None,
        ))
        step_tracking_frame(cfg, objects, t, dt)

    writer.release()

    objects_meta = [
        {
            "index": obj.index,
            "final_x": round(obj.x, 1),
            "final_y": round(obj.y, 1),
            "cued": obj.index in cued_indices,
        }
        for obj in objects
    ]

    ground_truth = {
        "trial_id": cfg.trial_id,
        "config": asdict(cfg),
        "video_path": video_path,
        "objects": objects_meta,
        "cued_indices": sorted(cued_indices),
        "probed_index": probed_index,
        "probe_is_target": probed_index in cued_indices,
        "t_probe_s": round(t_probe, 3),
    }

    gt_path = f"{out_dir}/ground_truth.json"
    with open(gt_path, "w") as f:
        json.dump(ground_truth, f, indent=2)

    return ground_truth
