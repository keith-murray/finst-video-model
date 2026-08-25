"""
Rigid-ring rotation for the debug-circular stimulus: n_objects crosses sit
equally spaced around one circle, starting from a randomly chosen angular
offset, and all rotate together by the same angle at every instant -- so
relative spacing (and therefore identity) is never ambiguous and, unlike
pylyshyn/smooth_pursuit, no minimum-separation resolution is needed.
"""

import math
import random

from finst_video_model.debug_circular.config import TrialConfig


class RingObject:
    def __init__(self, base_angle: float, index: int):
        self.base_angle = base_angle  # radians, position at frac=0
        self.index = index
        self.x = 0.0
        self.y = 0.0

    def place(self, cfg: TrialConfig, rotation_rad: float):
        cx, cy = cfg.image_width / 2, cfg.image_height / 2
        angle = self.base_angle + rotation_rad
        self.x = cx + cfg.circle_radius * math.cos(angle)
        self.y = cy + cfg.circle_radius * math.sin(angle)


def build_objects(cfg: TrialConfig) -> tuple[list["RingObject"], int]:
    """Places n_objects equally spaced around the circle starting from a
    randomly chosen offset, and picks the cued index."""
    placement_rng = random.Random(cfg.seed)
    start_offset = placement_rng.uniform(0, 2 * math.pi)
    spacing = 2 * math.pi / cfg.n_objects

    objects = [
        RingObject(base_angle=start_offset + i * spacing, index=i)
        for i in range(cfg.n_objects)
    ]
    for obj in objects:
        obj.place(cfg, rotation_rad=0.0)

    cue_rng = random.Random(cfg.seed + 1)
    cued_index = cue_rng.randrange(cfg.n_objects)

    return objects, cued_index


def step_tracking_frame(cfg: TrialConfig, objects: list["RingObject"], frac: float):
    """Places every object at its rigid-ring position for `frac` (0..1),
    the fraction of the way through the tracking phase's total rotation --
    frac=0 matches the cue phase's stationary positions exactly, frac=1
    reaches cfg.rotation_deg exactly, regardless of fps rounding.

    Screen space has y growing downward, so increasing angle in
    (cos(angle), sin(angle)) sweeps a point rightward -> downward ->
    leftward -> upward, i.e. clockwise as drawn -- so cfg.clockwise=True
    uses a positive angle and cfg.clockwise=False negates it.
    """
    total_rad = math.radians(cfg.rotation_deg)
    signed_total = total_rad if cfg.clockwise else -total_rad
    rotation_rad = signed_total * frac
    for obj in objects:
        obj.place(cfg, rotation_rad)
