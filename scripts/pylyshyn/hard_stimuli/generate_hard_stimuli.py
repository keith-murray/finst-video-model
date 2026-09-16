"""
Generates a 100-trial pylyshyn n_objects=3 stimulus set (fast redirect/fast
speed, n_cued=1) for `claude/2026_09/2026_09_16/TODO.md`'s "Part 1": test
many models on OpenRouter with the nearest-neighbor position heuristic
neutralized to chance, following up on
[[project-status-2026-09-14]]'s finding that heuristic-hard trials are
especially hard for gemma-4-31b-it (both locally and on OpenRouter).

Direct adaptation of
`generate_gemma_local_n4_heuristic_chance_stimuli.py`'s two-phase design at
n_objects=3 instead of 4 (see that script's docstring for the full
rationale) -- only OUT_ROOT/REPORT_PATH and N_OBJECTS differ. Two changes
since this run is OpenRouter-only (no local/cluster inference this time):
no `video.npy` is written in Phase 2 (only OpenRouter's chat-completions
API is used, which takes an mp4 directly), and Phase 2 additionally writes
`video_stretched.mp4` (same re-encode approach as
`run_pylyshyn_speed_sweep_models.py`: `write_mp4` at
`STRETCH_ENCODE_FPS = native_total_frames / STRETCH_TARGET_DURATION_S`),
since the sweep tests stretched video only.

Usage:
    uv run python scripts/pylyshyn/hard_stimuli/generate_hard_stimuli.py
"""

import json
import os

from finst_video_model.debug_circular.stimulus_gen import write_mp4
from finst_video_model.pylyshyn.config import TrialConfig
from finst_video_model.pylyshyn.stimulus_gen import generate_stimulus
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nearest_neighbor_heuristic import heuristic_accuracy_by_group, heuristic_predicts_match

N_OBJECTS = 3
N_CUED = 1
REDIRECT_S = 1.0
SPEED_MIN_PX_S = 32.0
SPEED_MAX_PX_S = 66.0

TARGET_PER_BUCKET = 25
MAX_SEEDS_SCANNED = 5000

STRETCH_TARGET_DURATION_S = 50.0
_NATIVE_TOTAL_FRAMES = TrialConfig(n_objects=N_OBJECTS, n_cued=N_CUED).total_frames
STRETCH_ENCODE_FPS = _NATIVE_TOTAL_FRAMES / STRETCH_TARGET_DURATION_S

OUT_ROOT = "data/pylyshyn/hard_all_models_sweep/trials"
REPORT_PATH = "results/pylyshyn/hard_stimuli/hard_all_models_sweep/generation_report.json"


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
    """Phase 1: scan seeds (in a throwaway tempdir, video.mp4 write only),
    return {(probe_is_target, heuristic_correct): [seed, ...]}."""
    import tempfile

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
    """Phase 2: regenerate the selected seeds for real, writing video.mp4
    (native) + video_stretched.mp4 + ground_truth.json."""
    os.makedirs(OUT_ROOT, exist_ok=True)
    rows = []

    for (probe_is_target_expected, heuristic_correct_expected), seeds in buckets.items():
        for seed in seeds:
            probe_on_target = (seed % 2 == 0)
            cfg = build_cfg(seed, probe_on_target)
            trial_dir = os.path.join(OUT_ROOT, cfg.trial_id)
            os.makedirs(trial_dir, exist_ok=True)

            ground_truth, frames = generate_stimulus(cfg, trial_dir)
            stretched_path = os.path.join(trial_dir, "video_stretched.mp4")
            write_mp4(frames, stretched_path, fps=STRETCH_ENCODE_FPS)

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
    print(f"n_objects={N_OBJECTS}. Native clip: {_NATIVE_TOTAL_FRAMES} frames "
          f"@ {TrialConfig().fps}fps. Stretched encode: {STRETCH_ENCODE_FPS:.2f}fps "
          f"(~{STRETCH_TARGET_DURATION_S:.0f}s nominal).")

    print(f"\nPhase 1: searching for {TARGET_PER_BUCKET} seeds per bucket "
          f"(4 buckets, 100 trials total)...")
    buckets = search_seeds()
    n_scanned_summary = {k: len(v) for k, v in buckets.items()}
    print(f"Done searching. Bucket sizes: {n_scanned_summary}")

    print("\nPhase 2: materializing selected trials (video.mp4 + video_stretched.mp4)...")
    rows = materialize(buckets)
    print(f"\nWrote {len(rows)} trials to {OUT_ROOT}")

    print("\nIndependent verification via heuristic_accuracy_by_group...")
    stats = heuristic_accuracy_by_group(rows, OUT_ROOT, lambda row: "n3")
    acc, sem, n = stats["n3"]
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
            "stretch_encode_fps": STRETCH_ENCODE_FPS,
        }, f, indent=2)
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
