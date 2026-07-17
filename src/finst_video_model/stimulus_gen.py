"""
Generates the single starting frame that gets fed to Veo 3.1 as the `image`
input, plus a ground-truth metadata JSON describing exactly which circles
were cued.

Only one frame is generated here — Veo never sees the tracking motion itself,
only this image plus the text prompt (see prompts.py). All motion, cueing,
de-cueing, and re-cueing is DESCRIBED in the prompt, not rendered by us.
"""

import json
import random
from dataclasses import asdict
from PIL import Image, ImageDraw

from config import TrialConfig


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
    """Writes `<out_dir>/frame0_<trial_id>.png` and
    `<out_dir>/ground_truth_<trial_id>.json`. Returns the ground truth dict.
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

    frame_path = f"{out_dir}/frame0_{cfg.trial_id}.png"
    img.save(frame_path)

    ground_truth = {
        "trial_id": cfg.trial_id,
        "config": asdict(cfg),
        "frame0_path": frame_path,
        "circles": circles_meta,
        "cued_indices": cued_indices,
    }

    gt_path = f"{out_dir}/ground_truth_{cfg.trial_id}.json"
    with open(gt_path, "w") as f:
        json.dump(ground_truth, f, indent=2)

    return ground_truth


if __name__ == "__main__":
    # Quick manual smoke test
    import os
    os.makedirs("stimuli", exist_ok=True)
    cfg = TrialConfig(n_circles=6, n_cued=2, seed=42)
    gt = generate_stimulus(cfg, "stimuli")
    print(f"Wrote {gt['frame0_path']}")
    print(f"Cued indices: {gt['cued_indices']}")
