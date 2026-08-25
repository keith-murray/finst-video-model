"""
Deterministic constant-velocity circle motion with elastic wall bounces.
Every position at every frame is exactly known, since we're generating the
ground truth, not inferring it after the fact.

A continuous minimum-separation constraint (`_resolve_separations`, applied
every tracking-phase frame via `step_tracking_frame`) now keeps any two
circles from passing through each other during motion -- matching
Pylyshyn's explicit no-identity-ambiguity guarantee, per
`claude/2026_08_14/TODO.md`'s "Updating the old task". `force_path_crossing`
(see `build_circles`) still biases cued circles' headings toward a close
encounter with the distractor crowd; the difference is that such encounters
now resolve as an elastic deflection rather than a visual overlap/pass-through.
"""

import math
import random

from finst_video_model.smooth_pursuit.config import TrialConfig


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


def build_circles(cfg: TrialConfig) -> tuple[list[Circle], set[int]]:
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


def _resolve_separations(circles: list[Circle], min_dist: float):
    """Elastic collision response (reflect the line-of-centers velocity
    component, push apart to exactly meet min_dist) for any pair of circles
    that end up closer than min_dist after a step -- keeps identity
    unambiguous during motion, not just at initial placement."""
    for i in range(len(circles)):
        for j in range(i + 1, len(circles)):
            a, b = circles[i], circles[j]
            dx, dy = b.x - a.x, b.y - a.y
            dist = math.hypot(dx, dy)
            if dist == 0 or dist >= min_dist:
                continue

            nx, ny = dx / dist, dy / dist

            overlap = min_dist - dist
            a.x -= nx * overlap / 2
            a.y -= ny * overlap / 2
            b.x += nx * overlap / 2
            b.y += ny * overlap / 2

            a_vn = a.vx * nx + a.vy * ny
            b_vn = b.vx * nx + b.vy * ny
            a.vx += (b_vn - a_vn) * nx
            a.vy += (b_vn - a_vn) * ny
            b.vx += (a_vn - b_vn) * nx
            b.vy += (a_vn - b_vn) * ny


def step_tracking_frame(cfg: TrialConfig, circles: list[Circle], dt: float):
    """Advances all circles by one frame during the tracking phase, then
    resolves any resulting close-approach violations."""
    for c in circles:
        c.step(dt, cfg.image_width, cfg.image_height)
    min_dist = cfg.circle_radius * cfg.min_center_distance_factor
    _resolve_separations(circles, min_dist)
