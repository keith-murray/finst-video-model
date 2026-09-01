"""
Simulates a "nearest neighbor" position heuristic for the pylyshyn task, per
`claude/2026_09/2026_09_01/TODO.md`'s Task 2: instead of tracking the cued
object's identity through the tracking-phase motion, a heuristic observer
just remembers where the (single, n_cued=1) cued object originally was, and
at probe time answers "yes, that's the target" iff the flashing probed
object is currently the closest of the n_objects to that remembered spot --
blind to which object is which, purely spatial.

Motion is fully deterministic given a trial's TrialConfig (all randomness is
seeded off cfg.seed, not off the trial's uuid `trial_id`), so this
re-simulates each trial's object positions from its saved
`ground_truth.json` (written by `finst_video_model.pylyshyn.stimulus_gen
.generate_stimulus`, one per trial dir under `data/pylyshyn/<run_name>/
trials/<trial_id>/`) without needing the actual video -- no OpenRouter
calls, no cost. Reuses `build_objects`/`step_tracking_frame` directly rather
than re-deriving the physics, so this can't drift out of sync with how the
real stimuli were generated. `probed_index`/`cued_indices`/`probe_is_target`
are trusted as already-saved ground truth (not re-derived) since they don't
depend on frame-by-frame position, only on cfg.seed via `_choose_probe`;
only the probe-time *positions* need re-simulating.

Because a trial's motion depends only on (seed, redirect_min_s/max_s,
speed_min/max_px_s, n_objects, n_cued) -- not on `variant` (native/
stretched only changes video encoding) or which model consumed it -- the
same heuristic accuracy applies to every model/variant that ran an
identical (redirect_condition, speed_condition, seed) trial, so this is
computed once per actual saved trial directory, not per model.
"""

import json
import math
import os

from finst_video_model.pylyshyn.config import TrialConfig
from finst_video_model.pylyshyn.physics import build_objects, step_tracking_frame


def heuristic_predicts_match(ground_truth: dict) -> bool:
    """True if the nearest-neighbor heuristic would answer "yes, this probe
    is the target" for this trial."""
    cfg = TrialConfig(**ground_truth["config"])
    objects, cued_indices = build_objects(cfg)
    assert sorted(cued_indices) == ground_truth["cued_indices"], (
        f"re-simulated cued_indices {sorted(cued_indices)} != saved "
        f"{ground_truth['cued_indices']} -- physics.py must have changed "
        f"since this trial was generated"
    )
    (cued_index,) = cued_indices  # n_cued=1 throughout this project's pylyshyn work
    anchor_x, anchor_y = next((o.x, o.y) for o in objects if o.index == cued_index)

    dt = 1.0 / cfg.fps
    probe_start_frame = int(round(ground_truth["t_probe_s"] * cfg.fps))
    for frame_i in range(probe_start_frame):
        step_tracking_frame(cfg, objects, t=frame_i * dt, dt=dt)

    nearest = min(objects, key=lambda o: math.hypot(o.x - anchor_x, o.y - anchor_y))
    return nearest.index == ground_truth["probed_index"]


def heuristic_accuracy_by_group(
    results_csv_rows: list[dict], data_trials_dir: str,
    group_key_fn,
) -> dict[tuple, tuple[float, float, int]]:
    """`results_csv_rows`: parsed rows (csv.DictReader dicts) from a
    results.csv sharing this project's `trial_id`/`probe_is_target` schema.
    `data_trials_dir`: the matching `data/pylyshyn/<run_name>/trials/` dir
    those trial_ids live under. `group_key_fn(row) -> tuple` picks the
    grouping key (e.g. (redirect_condition, speed_condition)).

    Returns {group_key: (accuracy_pct, sem_pct, n)}, deduplicating rows that
    share (group_key, seed, probe_on_target) -- native/stretched rows for
    the same seed are physically identical trials with different trial_ids,
    so only one is simulated per underlying trial to avoid double-counting
    it as independent evidence about the heuristic.
    """
    seen = set()
    totals = {}
    corrects = {}
    for row in results_csv_rows:
        group_key = group_key_fn(row)
        dedup_key = (group_key, int(row["seed"]), row["probe_on_target"] == "True")
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        gt_path = os.path.join(data_trials_dir, row["trial_id"], "ground_truth.json")
        with open(gt_path) as f:
            ground_truth = json.load(f)

        predicted_match = heuristic_predicts_match(ground_truth)
        correct = predicted_match == ground_truth["probe_is_target"]

        totals[group_key] = totals.get(group_key, 0) + 1
        corrects[group_key] = corrects.get(group_key, 0) + int(correct)

    stats = {}
    for key, n in totals.items():
        p = corrects[key] / n
        sem = math.sqrt(p * (1 - p) / n) * 100
        stats[key] = (p * 100, sem, n)
    return stats
