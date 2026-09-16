"""
Generates a 100-trial pylyshyn n_objects=4 stimulus set (fast redirect/fast
speed, native only, n_cued=1) -- see `claude/2026_09/2026_09_14/TODO.md`'s
Task 2. Follow-up to today's earlier finding that switching from video.npy
to native video.mp4 input dramatically improved local gemma-4-31b-it's
accuracy at n_objects=4 (65%->95%, see [[project-status-2026-09-14]]).
Two goals over the original 20-trial-per-condition set
(`generate_gemma_local_stimuli.py`): more power (50 target/50 distractor
instead of 10/10), and a sample deliberately constructed so the
nearest-neighbor position heuristic (`nearest_neighbor_heuristic.py`) scores
at chance (~50%) on it -- the original set's heuristic already scored above
chance at n=4 (75%), which muddies whether gemma's mp4 jump reflects real
tracking vs. exploiting the same positional regularity the heuristic
exploits.

Two-phase design, to avoid ever writing hundreds of unused ~44MB video.npy
files to disk for candidates that don't get selected:

  Phase 1 (search): scan seed=0,1,2,... indefinitely (capped at
  MAX_SEEDS_SCANNED), probe_on_target=(seed % 2 == 0) -- a parity-based
  50/50 split, since (unlike generate_gemma_local_stimuli.py's fixed
  20-seed list) the final pool size isn't known ahead of time so the
  existing "sort a fixed seed list, take the lower half" convention doesn't
  apply here. Each candidate is generated into a throwaway
  tempfile.TemporaryDirectory() (video.mp4 write is cheap, ~10KB; no
  video.npy). ground_truth.json is loaded and scored against
  `nearest_neighbor_heuristic.heuristic_predicts_match`, filing the seed
  into one of four (probe_is_target, heuristic_correct) buckets. A bucket
  stops accepting new seeds once it reaches TARGET_PER_BUCKET (25); the
  scan stops once all four buckets are full (100 total, heuristic accuracy
  on the union = exactly 50/100 = chance, by construction).

  Phase 2 (materialize): regenerates the 100 selected seeds for real into
  OUT_ROOT/<trial_id>/ with the usual video.mp4+video.npy+ground_truth.json
  triple (same shape as generate_gemma_local_stimuli.py's output), so the
  existing run_gemma_local_batch.py/run_gemma_local_batch_mp4.py/
  aggregate_gemma_local_*_nobjects_sweep.py scripts need zero changes to
  consume it -- just point --trials-root/--out-root at the new paths.
  Asserts each regenerated trial's probe_is_target/heuristic result matches
  what Phase 1 recorded for that seed (trial content is deterministic from
  cfg.seed, not cfg.trial_id -- confirmed by [[project-status-2026-09-10]]).

Finally, independently re-verifies the full 100-trial set's heuristic
accuracy via the actual production `heuristic_accuracy_by_group` function
(not just this script's own bucketing arithmetic) and writes a generation
report.

The user rsyncs OUT_ROOT to the cluster themselves afterward (same target
path convention as the other gemma_local_* trial roots:
scotty:/mnt/cup/people/km3199/finst-video-model/data/pylyshyn/
gemma_local_n4_heuristic_chance/trials/) -- code changes go through git
push/pull, not rsync, per [[feedback-git-for-code-rsync-for-data]].

Usage:
    uv run python scripts/pylyshyn/gemma_local/generate_gemma_local_n4_heuristic_chance_stimuli.py
"""

import json
import os
import tempfile

import numpy as np

from finst_video_model.pylyshyn.config import TrialConfig
from finst_video_model.pylyshyn.stimulus_gen import generate_stimulus
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nearest_neighbor_heuristic import heuristic_accuracy_by_group, heuristic_predicts_match

N_OBJECTS = 4
N_CUED = 1
REDIRECT_S = 1.0
SPEED_MIN_PX_S = 32.0
SPEED_MAX_PX_S = 66.0

TARGET_PER_BUCKET = 25
MAX_SEEDS_SCANNED = 5000

OUT_ROOT = "data/pylyshyn/gemma_local_n4_heuristic_chance/trials"
REPORT_PATH = "results/pylyshyn/gemma_local/gemma_local_n4_heuristic_chance/generation_report.json"


def build_cfg(seed: int, probe_on_target: bool) -> TrialConfig:
    return TrialConfig(
        n_objects=N_OBJECTS, n_cued=N_CUED, probe_on_target=probe_on_target, seed=seed,
        redirect_min_s=REDIRECT_S, redirect_max_s=REDIRECT_S,
        speed_min_px_s=SPEED_MIN_PX_S, speed_max_px_s=SPEED_MAX_PX_S,
    )


def bucket_full(buckets: dict, probe_on_target: bool) -> bool:
    return (
        len(buckets[(probe_on_target, True)]) >= TARGET_PER_BUCKET
        and len(buckets[(probe_on_target, False)]) >= TARGET_PER_BUCKET
    )


def search_seeds() -> dict:
    """Phase 1: scan seeds, return {(probe_is_target, heuristic_correct): [seed, ...]}."""
    buckets = {
        (True, True): [], (True, False): [],
        (False, True): [], (False, False): [],
    }

    for seed in range(MAX_SEEDS_SCANNED):
        if all(len(v) >= TARGET_PER_BUCKET for v in buckets.values()):
            break

        probe_on_target = (seed % 2 == 0)
        if bucket_full(buckets, probe_on_target):
            continue  # this class is already satisfied, skip generating it

        cfg = build_cfg(seed, probe_on_target)
        with tempfile.TemporaryDirectory() as tmp_dir:
            ground_truth, _frames = generate_stimulus(cfg, tmp_dir)

        probe_is_target = ground_truth["probe_is_target"]
        heuristic_correct = heuristic_predicts_match(ground_truth) == probe_is_target
        bucket_key = (probe_is_target, heuristic_correct)
        if len(buckets[bucket_key]) < TARGET_PER_BUCKET:
            buckets[bucket_key].append(seed)

        if (seed + 1) % 100 == 0:
            counts = {k: len(v) for k, v in buckets.items()}
            print(f"  scanned {seed + 1} seeds, bucket counts: {counts}")
    else:
        raise RuntimeError(
            f"scanned {MAX_SEEDS_SCANNED} seeds without filling all buckets: "
            f"{ {k: len(v) for k, v in buckets.items()} }"
        )

    return buckets


def materialize(buckets: dict) -> list[dict]:
    """Phase 2: regenerate the selected seeds for real, with video.npy this time."""
    os.makedirs(OUT_ROOT, exist_ok=True)
    rows = []

    for (probe_is_target_expected, heuristic_correct_expected), seeds in buckets.items():
        for seed in seeds:
            probe_on_target = (seed % 2 == 0)
            cfg = build_cfg(seed, probe_on_target)
            trial_dir = os.path.join(OUT_ROOT, cfg.trial_id)
            os.makedirs(trial_dir, exist_ok=True)

            ground_truth, frames = generate_stimulus(cfg, trial_dir)
            np.save(os.path.join(trial_dir, "video.npy"), frames)

            probe_is_target = ground_truth["probe_is_target"]
            heuristic_correct = heuristic_predicts_match(ground_truth) == probe_is_target
            assert probe_is_target == probe_is_target_expected, (
                f"seed={seed}: Phase 2 probe_is_target={probe_is_target} != "
                f"Phase 1's {probe_is_target_expected} -- generation is not deterministic"
            )
            assert heuristic_correct == heuristic_correct_expected, (
                f"seed={seed}: Phase 2 heuristic_correct={heuristic_correct} != "
                f"Phase 1's {heuristic_correct_expected} -- generation is not deterministic"
            )

            rows.append({
                "trial_id": cfg.trial_id,
                "seed": str(seed),
                "probe_on_target": str(probe_on_target),
            })
            print(
                f"  seed={seed} probe_on_target={probe_on_target} -> {trial_dir} "
                f"(probe_is_target={probe_is_target}, heuristic_correct={heuristic_correct})"
            )

    return rows


def main():
    print(f"Phase 1: searching for {TARGET_PER_BUCKET} seeds per bucket "
          f"(4 buckets, 100 trials total)...")
    buckets = search_seeds()
    n_scanned_summary = {k: len(v) for k, v in buckets.items()}
    print(f"Done searching. Bucket sizes: {n_scanned_summary}")

    print("\nPhase 2: materializing selected trials (video.mp4 + video.npy)...")
    rows = materialize(buckets)
    print(f"\nWrote {len(rows)} trials to {OUT_ROOT}")

    print("\nIndependent verification via heuristic_accuracy_by_group...")
    stats = heuristic_accuracy_by_group(rows, OUT_ROOT, lambda row: "n4")
    acc, sem, n = stats["n4"]
    print(f"Heuristic accuracy on the final {n}-trial set: {acc:.1f}% (sem={sem:.2f}%)")

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump({
            "n_objects": N_OBJECTS,
            "n_cued": N_CUED,
            "redirect_s": REDIRECT_S,
            "speed_min_px_s": SPEED_MIN_PX_S,
            "speed_max_px_s": SPEED_MAX_PX_S,
            "target_per_bucket": TARGET_PER_BUCKET,
            "n_trials": len(rows),
            "bucket_sizes": {str(k): v for k, v in n_scanned_summary.items()},
            "heuristic_accuracy_pct": acc,
            "heuristic_accuracy_sem_pct": sem,
        }, f, indent=2)
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
