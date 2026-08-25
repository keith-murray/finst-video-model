"""
Random-walk motion for the Pylyshyn reproduction, as opposed to
`finst_video_model.smooth_pursuit.physics`'s single constant heading per
object. Each object independently re-randomizes its direction (one of 8
compass headings) and speed once per redirect interval, elastically bounces
off walls every render frame, and is kept from ambiguous close approaches to
other objects via a continuous separation constraint.

The redirect interval is fixed at config.ASSUMED_VLM_SAMPLE_PERIOD_S (via
TrialConfig.redirect_min_s == redirect_max_s), not randomized on a
sub-second cadence: the downstream VLM only ever perceives two static
endpoint positions per sample interval, never the path between them, so the
net displacement it "sees" needs to be one explicit, controllable
(direction, speed) draw rather than an unpredictable sum of several
sub-second legs whose statistics would shift with the VLM's (unknown)
sampling phase.

The original paper enforces minimum separation by rejecting and
regenerating trajectory segments that violate it. We instead resolve
violations in place with an elastic collision response (reflect velocity
along the line of centers, push apart to restore the minimum distance) --
functionally the same guarantee (no ambiguous near-collisions), applied
continuously each frame rather than via rejection sampling, and consistent
with how wall bounces are already handled here.
"""

import math
import random

from finst_video_model.pylyshyn.config import TrialConfig

# 8 equal divisions of the compass, per the original.
_COMPASS_HEADINGS = [i * (2 * math.pi / 8) for i in range(8)]


class TrackedObject:
    def __init__(self, x, y, size, index, rng: random.Random):
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0
        self.size = size
        self.index = index
        self._rng = rng
        self.next_redirect_t = 0.0

    def _redirect(self, cfg: TrialConfig):
        angle = self._rng.choice(_COMPASS_HEADINGS)
        speed = self._rng.uniform(cfg.speed_min_px_s, cfg.speed_max_px_s)
        self.vx = speed * math.cos(angle)
        self.vy = speed * math.sin(angle)
        # redirect_min_s == redirect_max_s == ASSUMED_VLM_SAMPLE_PERIOD_S,
        # so this deterministically advances by exactly one VLM sample
        # period rather than a random sub-second interval.
        self.next_redirect_t += self._rng.uniform(cfg.redirect_min_s, cfg.redirect_max_s)

    def step(self, cfg: TrialConfig, t: float, dt: float):
        if t >= self.next_redirect_t:
            self._redirect(cfg)

        self.x += self.vx * dt
        self.y += self.vy * dt

        if self.x - self.size < 0:
            self.x = self.size + (self.size - self.x)
            self.vx *= -1
        elif self.x + self.size > cfg.image_width:
            overshoot = (self.x + self.size) - cfg.image_width
            self.x = cfg.image_width - self.size - overshoot
            self.vx *= -1

        if self.y - self.size < 0:
            self.y = self.size + (self.size - self.y)
            self.vy *= -1
        elif self.y + self.size > cfg.image_height:
            overshoot = (self.y + self.size) - cfg.image_height
            self.y = cfg.image_height - self.size - overshoot
            self.vy *= -1


def _resolve_separations(objects: list[TrackedObject], min_dist: float):
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            a, b = objects[i], objects[j]
            dx, dy = b.x - a.x, b.y - a.y
            dist = math.hypot(dx, dy)
            if dist == 0 or dist >= min_dist:
                continue

            nx, ny = dx / dist, dy / dist

            # Push apart along the normal so they exactly meet min_dist.
            overlap = min_dist - dist
            a.x -= nx * overlap / 2
            a.y -= ny * overlap / 2
            b.x += nx * overlap / 2
            b.y += ny * overlap / 2

            # Elastic collision (equal mass): swap the normal component of
            # velocity between the two objects.
            a_vn = a.vx * nx + a.vy * ny
            b_vn = b.vx * nx + b.vy * ny
            a.vx += (b_vn - a_vn) * nx
            a.vy += (b_vn - a_vn) * ny
            b.vx += (a_vn - b_vn) * nx
            b.vy += (a_vn - b_vn) * ny


def _sample_non_overlapping_positions(cfg: TrialConfig, rng: random.Random):
    min_dist = cfg.object_size * cfg.min_center_distance_factor
    margin = cfg.object_size * 2
    positions = []
    attempts = 0
    while len(positions) < cfg.n_objects and attempts < 20000:
        attempts += 1
        x = rng.uniform(margin, cfg.image_width - margin)
        y = rng.uniform(margin, cfg.image_height - margin)
        if all((x - px) ** 2 + (y - py) ** 2 >= min_dist ** 2 for px, py in positions):
            positions.append((x, y))
    if len(positions) < cfg.n_objects:
        raise RuntimeError(
            f"Could not place {cfg.n_objects} non-overlapping objects. "
            f"Reduce n_objects/object_size or increase image size."
        )
    return positions


def build_objects(cfg: TrialConfig) -> tuple[list[TrackedObject], set[int]]:
    """Creates objects at random non-overlapping start positions (stationary
    until the tracking phase begins) and picks the cued subset."""
    placement_rng = random.Random(cfg.seed)
    positions = _sample_non_overlapping_positions(cfg, placement_rng)

    cue_rng = random.Random(cfg.seed + 1)
    cued_indices = set(cue_rng.sample(range(cfg.n_objects), cfg.n_cued))

    objects = []
    for i, (x, y) in enumerate(positions):
        object_rng = random.Random(cfg.seed + 100 + i)
        objects.append(TrackedObject(x, y, cfg.object_size, i, object_rng))

    return objects, cued_indices


def step_tracking_frame(cfg: TrialConfig, objects: list[TrackedObject], t: float, dt: float):
    """Advances all objects by one frame during the tracking phase: each
    object independently redirects/moves/bounces, then any resulting
    close-approach violations are resolved."""
    for obj in objects:
        obj.step(cfg, t, dt)
    min_dist = cfg.object_size * cfg.min_center_distance_factor
    _resolve_separations(objects, min_dist)
