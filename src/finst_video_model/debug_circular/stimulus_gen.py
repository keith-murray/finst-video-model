"""
Renders the complete trial video for the debug-circular stimulus -- the
simplest possible motion-perception check ahead of returning to the
pylyshyn/smooth_pursuit arms (see `claude/2026_08/2026_08_25/TODO.md`):
  1. Cue phase (cfg.cue_flash_s): objects stationary, equally spaced around
     the circle. One (cued_index) is drawn red; the rest are white crosses.
  2. Tracking phase (cfg.tracking_s): all objects rotate together as a
     rigid ring (see `physics.py`) by cfg.rotation_deg total, in the
     direction given by cfg.clockwise, at constant angular velocity. All
     crosses stay white during this phase -- identity must be tracked
     purely by motion, not color.
  3. Probe phase (cfg.probe_flash_s): motion freezes at its final rotated
     position, and exactly one object (probed_index, chosen per
     cfg.probe_on_target) is drawn red; the model is asked a single
     True/False question -- "was this the object cued at the start?"

Ground truth (every object's base angle, which index was cued, and which
was probed) is known exactly, since we simulate every position ourselves.

Always writes `<out_dir>/video.npy` (every rendered frame stacked into one
`(n_frames, H, W, 3)` uint8 RGB array -- what the locally-hosted
Qwen3.8-27B actually receives) and `<out_dir>/ground_truth.json`. Also
writes `<out_dir>/video.mp4` (for human eyeballing/debugging) unless
`save_mp4=False` -- the cluster's compute nodes have no GPU video-encode
device and can't reliably produce the mp4 (see
`claude/skills/cluster/qwen38_cluster_handoff.md`'s "skip video encoding
entirely for synthetic data" note), so stimuli generated directly on the
cluster should pass `save_mp4=False` / `--no-save-mp4`.
"""

import json
import math
import random
from dataclasses import asdict

import cv2
import numpy as np

from finst_video_model.debug_circular.config import TrialConfig
from finst_video_model.debug_circular.physics import build_objects, step_tracking_frame


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


def _draw_frame(cfg: TrialConfig, objects, highlighted_index: int | None) -> np.ndarray:
    img = np.full((cfg.image_height, cfg.image_width, 3), cfg.background_color, dtype=np.uint8)
    for obj in objects:
        color = cfg.cued_color if obj.index == highlighted_index else cfg.neutral_color
        _draw_cross(img, obj.x, obj.y, cfg.object_size, cfg.arm_thickness, color)
    return img


def _choose_probe(cfg: TrialConfig, cued_index: int, rng: random.Random) -> int:
    if cfg.probe_on_target:
        return cued_index
    return rng.choice([i for i in range(cfg.n_objects) if i != cued_index])


def generate_stimulus(cfg: TrialConfig, out_dir: str, save_mp4: bool = True) -> dict:
    cfg.validate()

    objects, cued_index = build_objects(cfg)
    probe_rng = random.Random(cfg.seed + 2)
    probed_index = _choose_probe(cfg, cued_index, probe_rng)

    n_cue_frames = int(round(cfg.cue_flash_s * cfg.fps))
    n_track_frames = int(round(cfg.tracking_s * cfg.fps))
    n_probe_frames = int(round(cfg.probe_flash_s * cfg.fps))

    video_path = f"{out_dir}/video.mp4" if save_mp4 else None
    writer = None
    if save_mp4:
        # avc1 (H.264) rather than mp4v -- mp4v renders as solid green in
        # QuickTime/macOS's default player even though the pixel data is fine.
        writer = cv2.VideoWriter(
            video_path, cv2.VideoWriter_fourcc(*"avc1"), cfg.fps,
            (cfg.image_width, cfg.image_height)
        )
    frames = []

    def _write(frame: np.ndarray):
        if writer is not None:
            writer.write(frame)
        frames.append(frame[..., ::-1])  # BGR (cv2 convention) -> RGB

    # --- Cue phase: stationary, cued index shown red ---
    for _ in range(n_cue_frames):
        _write(_draw_frame(cfg, objects, highlighted_index=cued_index))

    # --- Tracking phase: rigid ring rotation, all white ---
    for frame_i in range(n_track_frames):
        frac = frame_i / (n_track_frames - 1) if n_track_frames > 1 else 1.0
        step_tracking_frame(cfg, objects, frac)
        _write(_draw_frame(cfg, objects, highlighted_index=None))

    # --- Probe phase: frozen at final position, probed index shown red ---
    probe_frame = _draw_frame(cfg, objects, highlighted_index=probed_index)
    for _ in range(n_probe_frames):
        _write(probe_frame)

    if writer is not None:
        writer.release()

    npy_path = f"{out_dir}/video.npy"
    np.save(npy_path, np.stack(frames, axis=0))

    objects_meta = [
        {
            "index": obj.index,
            "base_angle_deg": round(math.degrees(obj.base_angle) % 360, 2),
            "final_x": round(obj.x, 1),
            "final_y": round(obj.y, 1),
            "cued": obj.index == cued_index,
        }
        for obj in objects
    ]

    ground_truth = {
        "trial_id": cfg.trial_id,
        "config": asdict(cfg),
        "video_path": video_path,
        "npy_path": npy_path,
        "objects": objects_meta,
        "cued_index": cued_index,
        "probed_index": probed_index,
        "probe_is_target": probed_index == cued_index,
    }

    gt_path = f"{out_dir}/ground_truth.json"
    with open(gt_path, "w") as f:
        json.dump(ground_truth, f, indent=2)

    return ground_truth
