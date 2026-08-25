"""
Deterministic constant-velocity circle motion with elastic wall bounces.
No circle-circle collision physics -- circles may visually overlap/cross,
which is intentional (crossing events are the MOT-relevant manipulation,
not a bug to fix). Every position at every frame is exactly known, since
we're generating the ground truth, not inferring it after the fact.
"""

import math
import random
from config import TrialConfig


class Circle:
    def __init__(self, x, y, vx, vy, radius, index):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.radius = radius
        self.index = index

    def step(self, dt, width, height):
        self.x += self.vx * dt
        self.y += self.vy * dt

        # Elastic bounce off walls
        if self.x - self.radius < 0:
            self.x = self.radius + (self.radius - self.x)
            self.vx *= -1
        elif self.x + self.radius > width:
            overshoot = (self.x + self.radius) - width
            self.x = width - self.radius - overshoot
            self.vx *= -1

        if self.y - self.radius < 0:
            self.y = self.radius + (self.radius - self.y)
            self.vy *= -1
        elif self.y + self.radius > height:
            overshoot = (self.y + self.radius) - height
            self.y = height - self.radius - overshoot
            self.vy *= -1


def _sample_non_overlapping_positions(cfg: TrialConfig, rng: random.Random):
    min_dist = cfg.circle_radius * cfg.min_center_distance_factor
    margin = cfg.circle_radius * 2
    positions = []
    attempts = 0
    while len(positions) < cfg.n_circles and attempts < 20000:
        attempts += 1
        x = rng.uniform(margin, cfg.image_width - margin)
        y = rng.uniform(margin, cfg.image_height - margin)
        if all((x - px) ** 2 + (y - py) ** 2 >= min_dist ** 2 for px, py in positions):
            positions.append((x, y))
    if len(positions) < cfg.n_circles:
        raise RuntimeError(
            f"Could not place {cfg.n_circles} non-overlapping circles. "
            f"Reduce n_circles/radius or increase image size."
        )
    return positions


def build_circles(cfg: TrialConfig) -> list[Circle]:
    """Creates circles with random non-overlapping start positions and random
    initial headings, all at the same constant speed. If
    cfg.force_path_crossing, cued circles' initial headings are biased to
    point roughly toward the centroid of the distractors, to encourage
    trajectory crossings during the tracking phase."""
    rng = random.Random(cfg.seed)
    positions = _sample_non_overlapping_positions(cfg, rng)

    cue_rng = random.Random(cfg.seed + 1)
    cued_indices = set(cue_rng.sample(range(cfg.n_circles), cfg.n_cued))

    circles = []
    cx_all = sum(p[0] for p in positions) / len(positions)
    cy_all = sum(p[1] for p in positions) / len(positions)

    for i, (x, y) in enumerate(positions):
        angle_rng = random.Random(cfg.seed + 100 + i)
        if cfg.force_path_crossing and i in cued_indices:
            # Aim roughly at the opposite side of the display (through the
            # crowd of distractors) with some jitter, rather than a fully
            # random heading.
            base_angle = math.atan2(cy_all - y, cx_all - x)
            angle = base_angle + angle_rng.uniform(-0.4, 0.4)
        else:
            angle = angle_rng.uniform(0, 2 * math.pi)

        vx = cfg.speed_px_s * math.cos(angle)
        vy = cfg.speed_px_s * math.sin(angle)
        circles.append(Circle(x, y, vx, vy, cfg.circle_radius, i))

    return circles, cued_indices
