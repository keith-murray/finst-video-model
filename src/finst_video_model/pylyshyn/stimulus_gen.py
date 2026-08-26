"""
Renders the complete trial video for the Pylyshyn reproduction:
  1. Cue phase (cfg.cue_s): all objects stationary, drawn as crosses. The
     cued subset is drawn steadily in cfg.cued_color (red); distractors stay
     cfg.neutral_color (white) -- no blinking (see config.py's "Part 3" note:
     visual design ported from debug_circular).
  2. Tracking phase (cfg.tracking_s): all objects move identically (no
     color distinction -- cueing information must be tracked purely by
     motion from here on), via continuously-redirecting random-walk physics
     (see `physics.py`, unchanged). At a single randomly chosen moment
     within this phase, one object -- either a cued target or a distractor,
     depending on cfg.probe_on_target -- is drawn in cfg.cued_color (still a
     cross, not a different shape) for cfg.probe_flash_s, then reverts to
     white and motion continues to the end of the phase.

Ground truth (which objects were cued, which one was probed, and whether
that probe was a target) is known exactly, since we simulated every
position ourselves.

Writes `<out_dir>/video.mp4` and `<out_dir>/ground_truth.json`, matching the
convention used by the other arms. Also returns the rendered RGB frame
array in-memory (not persisted to disk -- at cfg.fps=24/~18s this would be
~190MB/trial as a raw .npy, unlike debug_circular's much shorter clips)
so a caller can immediately re-encode a duration-stretched mp4 variant from
the same lossless source (see scripts/pylyshyn/run_pylyshyn_stretch_sweep.py
and finst_video_model.debug_circular.stimulus_gen.write_mp4) without a
lossy decode-recompress round trip through the already-compressed native
video.mp4.
"""

import json
import random
from dataclasses import asdict

import cv2
import numpy as np

from finst_video_model.pylyshyn.config import TrialConfig
from finst_video_model.pylyshyn.physics import build_objects, step_tracking_frame


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


def _draw_frame(cfg: TrialConfig, objects, highlighted_indices: set[int]) -> np.ndarray:
    img = np.full((cfg.image_height, cfg.image_width, 3), cfg.background_color, dtype=np.uint8)
    for obj in objects:
        color = cfg.cued_color if obj.index in highlighted_indices else cfg.neutral_color
        _draw_cross(img, obj.x, obj.y, cfg.object_size, cfg.arm_thickness, color)
    return img


def _choose_probe(cfg: TrialConfig, cued_indices: set[int], rng: random.Random):
    candidates = sorted(cued_indices if cfg.probe_on_target else set(range(cfg.n_objects)) - cued_indices)
    probed_index = rng.choice(candidates)
    t_probe = rng.uniform(cfg.min_probe_delay_s, cfg.tracking_s - cfg.min_post_probe_s)
    return probed_index, t_probe


def generate_stimulus(cfg: TrialConfig, out_dir: str) -> tuple[dict, np.ndarray]:
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
    frames = []

    def _write(frame: np.ndarray):
        writer.write(frame)
        frames.append(frame[..., ::-1])  # BGR (cv2 convention) -> RGB

    # --- Cue phase: stationary, cued subset shown red (steady, no blink) ---
    for _ in range(n_cue_frames):
        _write(_draw_frame(cfg, objects, highlighted_indices=cued_indices))

    # --- Tracking phase: all objects move; one probe flash mid-phase ---
    for frame_i in range(n_track_frames):
        t = frame_i * dt
        is_probing = probe_start_frame <= frame_i < probe_start_frame + n_probe_frames
        _write(_draw_frame(
            cfg, objects, highlighted_indices={probed_index} if is_probing else set(),
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

    return ground_truth, np.stack(frames, axis=0)
