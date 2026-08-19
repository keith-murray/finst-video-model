"""
Renders the complete trial video:
  1. Cue phase (cfg.cue_flash_s): K circles red, rest gray, all moving.
  2. Tracking phase (cfg.tracking_s): all circles identical gray, moving.
  3. Label phase (cfg.label_s): motion freezes at its final position, and
     each circle gets a letter label overlaid (A, B, C, ...). No recoloring
     happens -- identity must be reported via the letters, keeping scoring
     a simple text-match problem instead of a vision problem.

Ground truth (which letters correspond to originally-cued circles) is known
exactly, since we simulated every position ourselves.
"""

import json
import string
import cv2
import numpy as np
from dataclasses import asdict

from config import TrialConfig
from physics import build_circles


def _draw_frame(cfg: TrialConfig, circles, cued_indices, phase: str,
                 label_map: dict | None = None) -> np.ndarray:
    img = np.full((cfg.image_height, cfg.image_width, 3), cfg.background_color, dtype=np.uint8)

    for c in circles:
        if phase == "cue":
            color = cfg.cued_color if c.index in cued_indices else cfg.neutral_color
        else:
            color = cfg.neutral_color

        center = (int(round(c.x)), int(round(c.y)))
        cv2.circle(img, center, cfg.circle_radius, color[::-1], thickness=-1)  # BGR
        cv2.circle(img, center, cfg.circle_radius, (0, 0, 0), thickness=2)

        if phase == "label" and label_map is not None:
            letter = label_map[c.index]
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(letter, font, 0.9, 2)
            text_org = (center[0] - tw // 2, center[1] + th // 2)
            cv2.putText(img, letter, text_org, font, 0.9,
                        cfg.label_text_color[::-1], 2, cv2.LINE_AA)

    return img


def generate_stimulus(cfg: TrialConfig, out_dir: str) -> dict:
    cfg.validate()

    circles, cued_indices = build_circles(cfg)
    dt = 1.0 / cfg.fps

    n_cue_frames = int(round(cfg.cue_flash_s * cfg.fps))
    n_track_frames = int(round(cfg.tracking_s * cfg.fps))
    n_label_frames = int(round(cfg.label_s * cfg.fps))

    video_path = f"{out_dir}/video_{cfg.trial_id}.mp4"
    writer = cv2.VideoWriter(
        video_path, cv2.VideoWriter_fourcc(*"mp4v"), cfg.fps,
        (cfg.image_width, cfg.image_height)
    )

    # --- Cue phase ---
    for _ in range(n_cue_frames):
        writer.write(_draw_frame(cfg, circles, cued_indices, "cue"))
        for c in circles:
            c.step(dt, cfg.image_width, cfg.image_height)

    # --- Tracking phase ---
    for _ in range(n_track_frames):
        writer.write(_draw_frame(cfg, circles, cued_indices, "track"))
        for c in circles:
            c.step(dt, cfg.image_width, cfg.image_height)

    # --- Label phase: freeze motion, assign letters, hold ---
    letters = list(string.ascii_uppercase[:cfg.n_circles])
    label_rng_seed = cfg.seed + 999
    import random
    random.Random(label_rng_seed).shuffle(letters)
    label_map = {c.index: letters[i] for i, c in enumerate(circles)}

    label_frame = _draw_frame(cfg, circles, cued_indices, "label", label_map)
    for _ in range(n_label_frames):
        writer.write(label_frame)

    writer.release()

    circles_meta = [
        {
            "index": c.index,
            "final_x": round(c.x, 1),
            "final_y": round(c.y, 1),
            "label": label_map[c.index],
            "cued": c.index in cued_indices,
        }
        for c in circles
    ]
    cued_letters = sorted(label_map[i] for i in cued_indices)

    ground_truth = {
        "trial_id": cfg.trial_id,
        "config": asdict(cfg),
        "video_path": video_path,
        "circles": circles_meta,
        "cued_indices": sorted(cued_indices),
        "cued_letters": cued_letters,
    }

    gt_path = f"{out_dir}/ground_truth_{cfg.trial_id}.json"
    with open(gt_path, "w") as f:
        json.dump(ground_truth, f, indent=2)

    return ground_truth


if __name__ == "__main__":
    import os
    os.makedirs("stimuli", exist_ok=True)
    cfg = TrialConfig(n_circles=6, n_cued=2, seed=42,
                       cue_flash_s=1.0, tracking_s=6.0, label_s=2.0)
    gt = generate_stimulus(cfg, "stimuli")
    print(f"Wrote {gt['video_path']}")
    print(f"Cued letters (ground truth answer): {gt['cued_letters']}")
