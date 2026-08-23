"""
Generates a handful of sample smooth-pursuit videos so the reformatted
stimulus (stationary cue phase, continuously-separation-constrained
tracking motion, end-of-trial probe highlight) can be eyeballed before any
VLM is wired up.

Sweeps n_cued and probe_on_target (the two conditions needed for a d'
computation later) across a few seeds each, writing every trial to its own
`data/smooth_pursuit_samples/<trial_id>/` directory.

Usage:
    uv run python scripts/smooth_pursuit/generate_samples.py
    uv run python scripts/smooth_pursuit/generate_samples.py --n-cued 1 2 4 --seeds-per-condition 2
"""

import argparse
import os

from finst_video_model.comprehension.smooth_pursuit.config import TrialConfig
from finst_video_model.comprehension.smooth_pursuit.stimulus_gen import generate_stimulus


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-cued", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--seeds-per-condition", type=int, default=1)
    parser.add_argument("--out-root", type=str, default="data/smooth_pursuit_samples")
    parser.add_argument(
        "--save-npy", action="store_true",
        help="Also save each trial's raw frames as <trial_dir>/video.npy "
             "(uint8 RGB, one array per trial) alongside video.mp4, for "
             "comparing against the mp4 to check for encoding artifacts.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_root, exist_ok=True)

    conditions = [
        (n_cued, probe_on_target)
        for n_cued in args.n_cued
        for probe_on_target in (True, False)
    ]

    for n_cued, probe_on_target in conditions:
        for rep in range(args.seeds_per_condition):
            seed = n_cued * 1000 + (0 if probe_on_target else 500) + rep
            cfg = TrialConfig(n_cued=n_cued, probe_on_target=probe_on_target, seed=seed)
            trial_dir = os.path.join(args.out_root, cfg.trial_id)
            os.makedirs(trial_dir, exist_ok=True)

            ground_truth = generate_stimulus(cfg, trial_dir, save_npy=args.save_npy)

            print(
                f"n_cued={n_cued} probe_on_target={probe_on_target} seed={seed} "
                f"-> {trial_dir} (probed_index={ground_truth['probed_index']}, "
                f"probe_is_target={ground_truth['probe_is_target']})"
            )


if __name__ == "__main__":
    main()
