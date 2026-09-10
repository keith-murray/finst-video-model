"""
Generates the pylyshyn n_objects={3,4,5} x seeds=0-19 stimulus set (fast
redirect/fast speed, native only, n_cued=1) for the locally-hosted
gemma-4-31b-it cluster run -- see `claude/2026_09/2026_09_10/TODO.md`.

Writes each trial to its own
`data/pylyshyn/gemma_local_nobjects_sweep/trials/<trial_id>/`, with the same
`video.mp4`/`ground_truth.json` pair every other pylyshyn script writes, plus
a `video.npy` (raw RGB frame stack) for `run_gemma_local_batch.py` (the
cluster script) to consume directly -- mirrors debug_circular's local-
cluster data layout, since `finst_video_model.pylyshyn.stimulus_gen
.generate_stimulus` (unlike debug_circular's) doesn't persist this itself.

`TrialConfig.trial_id` defaults to a random uuid4 per instantiation, not a
hash of config fields -- so this run's trial directories are NOT the same
ones OpenRouter's `speed_sweep_models`/`nobjects_sweep_n4`/`nobjects_sweep_n5`
runs used, even at identical seeds. That's fine: trial content (motion,
ground truth) is fully deterministic from `cfg.seed`, so this run's trial at
a given (n_objects, seed) has identical ground truth to OpenRouter's trial
at that same (n_objects, seed) despite living in a new directory.

probe_on_target split matches `run_pylyshyn_speed_sweep_models.py`'s
`build_grid`: the lower half of sorted seeds gets probe_on_target=True.

Usage:
    uv run python scripts/pylyshyn/generate_gemma_local_stimuli.py
"""

import os

import numpy as np

from finst_video_model.pylyshyn.config import TrialConfig
from finst_video_model.pylyshyn.stimulus_gen import generate_stimulus

N_OBJECTS_VALUES = [3, 4, 5]
N_CUED = 1
SEEDS = list(range(20))
REDIRECT_S = 1.0
SPEED_MIN_PX_S = 32.0
SPEED_MAX_PX_S = 66.0
OUT_ROOT = "data/pylyshyn/gemma_local_nobjects_sweep/trials"


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)

    half = len(SEEDS) // 2
    probe_on_target_by_seed = {seed: (i < half) for i, seed in enumerate(sorted(SEEDS))}

    for n_objects in N_OBJECTS_VALUES:
        for seed in sorted(SEEDS):
            probe_on_target = probe_on_target_by_seed[seed]
            cfg = TrialConfig(
                n_objects=n_objects, n_cued=N_CUED, probe_on_target=probe_on_target, seed=seed,
                redirect_min_s=REDIRECT_S, redirect_max_s=REDIRECT_S,
                speed_min_px_s=SPEED_MIN_PX_S, speed_max_px_s=SPEED_MAX_PX_S,
            )
            trial_dir = os.path.join(OUT_ROOT, cfg.trial_id)
            os.makedirs(trial_dir, exist_ok=True)

            ground_truth, frames = generate_stimulus(cfg, trial_dir)
            np.save(os.path.join(trial_dir, "video.npy"), frames)

            print(
                f"n_objects={n_objects} seed={seed} probe_on_target={probe_on_target} "
                f"-> {trial_dir} (probe_is_target={ground_truth['probe_is_target']}, "
                f"frames={frames.shape})"
            )

    print(f"\nWrote {len(N_OBJECTS_VALUES) * len(SEEDS)} trials to {OUT_ROOT}")


if __name__ == "__main__":
    main()
