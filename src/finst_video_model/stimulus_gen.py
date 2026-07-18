"""
Generates the single starting frame that gets fed to Veo 3.1 as the `image`
input, plus a ground-truth metadata JSON describing exactly which circles
were cued.

Only one frame is generated here — Veo never sees the tracking motion itself,
only this image plus the text prompt (see prompts.py). All motion, cueing,
de-cueing, and re-cueing is DESCRIBED in the prompt, not rendered by us.
"""

import json
import math
import random
from dataclasses import asdict
from PIL import Image, ImageDraw

from finst_video_model.config import TrialConfig


def _sample_non_overlapping_centers(cfg: TrialConfig) -> list[tuple[int, int]]:
    """Rejection-sample circle centers with margin so circles are far enough
    apart to be visually unambiguous in the first frame, and don't touch the
    image border."""
    rng = random.Random(cfg.seed)
    min_dist = cfg.circle_radius * cfg.min_center_distance_factor
    margin = cfg.circle_radius * 2

    centers: list[tuple[int, int]] = []
    max_attempts = 20000
    attempts = 0

    while len(centers) < cfg.n_circles and attempts < max_attempts:
        attempts += 1
        x = rng.uniform(margin, cfg.image_width - margin)
        y = rng.uniform(margin, cfg.image_height - margin)

        if all((x - cx) ** 2 + (y - cy) ** 2 >= min_dist ** 2 for cx, cy in centers):
            centers.append((x, y))

    if len(centers) < cfg.n_circles:
        raise RuntimeError(
            f"Could not place {cfg.n_circles} non-overlapping circles in a "
            f"{cfg.image_width}x{cfg.image_height} frame with radius "
            f"{cfg.circle_radius}. Reduce n_circles, reduce circle_radius, "
            f"or increase image size."
        )
    return centers


def generate_stimulus(cfg: TrialConfig, out_dir: str) -> dict:
    """Writes `<out_dir>/frame0.png` and `<out_dir>/ground_truth.json`.
    `out_dir` is expected to already be trial-specific (e.g.
    `data/<trial_id>/`). Returns the ground truth dict.
    """
    cfg.validate()

    centers = _sample_non_overlapping_centers(cfg)

    rng = random.Random(cfg.seed + 1)  # separate stream for cue assignment
    cued_indices = sorted(rng.sample(range(cfg.n_circles), cfg.n_cued))

    img = Image.new("RGB", (cfg.image_width, cfg.image_height), cfg.background_color)
    draw = ImageDraw.Draw(img)

    circles_meta = []
    for i, (x, y) in enumerate(centers):
        is_cued = i in cued_indices
        color = cfg.cued_color if is_cued else cfg.neutral_color
        bbox = [x - cfg.circle_radius, y - cfg.circle_radius,
                 x + cfg.circle_radius, y + cfg.circle_radius]
        draw.ellipse(bbox, fill=color, outline=(0, 0, 0), width=2)

        circles_meta.append({
            "index": i,
            "center_x": round(x, 1),
            "center_y": round(y, 1),
            "radius": cfg.circle_radius,
            "cued": is_cued,
        })

    frame_path = f"{out_dir}/frame0.png"
    img.save(frame_path)

    ground_truth = {
        "trial_id": cfg.trial_id,
        "config": asdict(cfg),
        "frame0_path": frame_path,
        "circles": circles_meta,
        "cued_indices": cued_indices,
    }

    gt_path = f"{out_dir}/ground_truth.json"
    with open(gt_path, "w") as f:
        json.dump(ground_truth, f, indent=2)

    return ground_truth


def generate_circular_track_stimulus(cfg: TrialConfig, out_dir: str) -> dict:
    """Renders circles evenly spaced around a large circular track drawn in
    the frame, rather than randomly scattered. The track itself gives Veo an
    explicit path to follow, constraining the motion far more than a free-
    form physics description can.

    Writes `<out_dir>/frame0.png` and `<out_dir>/ground_truth.json`, same as
    `generate_stimulus`. `out_dir` is expected to already be trial-specific.
    """
    cfg.validate()

    cx, cy = cfg.image_width / 2, cfg.image_height / 2
    track_radius = cfg.track_radius

    margin = cfg.circle_radius * 2
    max_track_radius = min(cfg.image_width, cfg.image_height) / 2 - cfg.circle_radius - margin
    assert track_radius <= max_track_radius, (
        f"track_radius ({track_radius}) too large for a {cfg.image_width}x"
        f"{cfg.image_height} frame with circle_radius {cfg.circle_radius}; "
        f"must be <= {max_track_radius:.0f}"
    )

    adjacent_chord = 2 * track_radius * math.sin(math.pi / cfg.n_circles)
    min_dist = cfg.circle_radius * cfg.min_center_distance_factor
    assert adjacent_chord >= min_dist, (
        f"{cfg.n_circles} circles evenly spaced on a track of radius "
        f"{track_radius} would sit too close together (chord distance "
        f"{adjacent_chord:.1f} < {min_dist:.1f}); increase track_radius or "
        f"reduce n_circles."
    )

    rng = random.Random(cfg.seed + 1)  # same stream convention as generate_stimulus
    cued_indices = sorted(rng.sample(range(cfg.n_circles), cfg.n_cued))

    img = Image.new("RGB", (cfg.image_width, cfg.image_height), cfg.background_color)
    draw = ImageDraw.Draw(img)

    track_bbox = [cx - track_radius, cy - track_radius, cx + track_radius, cy + track_radius]
    draw.ellipse(track_bbox, outline=cfg.track_color, width=cfg.track_line_width)

    circles_meta = []
    for i in range(cfg.n_circles):
        angle_deg = cfg.track_start_angle_deg + i * (360.0 / cfg.n_circles)
        angle_rad = math.radians(angle_deg)
        x = cx + track_radius * math.cos(angle_rad)
        y = cy + track_radius * math.sin(angle_rad)

        is_cued = i in cued_indices
        color = cfg.cued_color if is_cued else cfg.neutral_color
        bbox = [x - cfg.circle_radius, y - cfg.circle_radius,
                 x + cfg.circle_radius, y + cfg.circle_radius]
        draw.ellipse(bbox, fill=color, outline=(0, 0, 0), width=2)

        circles_meta.append({
            "index": i,
            "angle_deg": round(angle_deg % 360, 1),
            "center_x": round(x, 1),
            "center_y": round(y, 1),
            "radius": cfg.circle_radius,
            "cued": is_cued,
        })

    frame_path = f"{out_dir}/frame0.png"
    img.save(frame_path)

    ground_truth = {
        "trial_id": cfg.trial_id,
        "config": asdict(cfg),
        "frame0_path": frame_path,
        "track_center": [cx, cy],
        "track_radius": track_radius,
        "circles": circles_meta,
        "cued_indices": cued_indices,
    }

    gt_path = f"{out_dir}/ground_truth.json"
    with open(gt_path, "w") as f:
        json.dump(ground_truth, f, indent=2)

    return ground_truth


if __name__ == "__main__":
    # Quick manual smoke test
    import os
    cfg = TrialConfig(n_circles=6, n_cued=2, seed=42)
    out_dir = f"data/{cfg.trial_id}"
    os.makedirs(out_dir, exist_ok=True)
    gt = generate_stimulus(cfg, out_dir)
    print(f"Wrote {gt['frame0_path']}")
    print(f"Cued indices: {gt['cued_indices']}")

    track_cfg = TrialConfig(n_circles=3, n_cued=1, seed=42)
    track_out_dir = f"data/{track_cfg.trial_id}"
    os.makedirs(track_out_dir, exist_ok=True)
    track_gt = generate_circular_track_stimulus(track_cfg, track_out_dir)
    print(f"Wrote {track_gt['frame0_path']}")
    print(f"Cued indices: {track_gt['cued_indices']}")
