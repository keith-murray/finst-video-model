"""
Generates a handful of sample Pylyshyn-reproduction videos so the stimulus
(blinking cue phase, random-walk tracking motion, mid-trial probe flash) can
be eyeballed before any VLM is wired up.

Sweeps n_cued and probe_on_target (the two conditions needed for a d'
computation later) across a few seeds each, writing every trial to its own
`data/pylyshyn_samples/<trial_id>/` directory.

Usage:
    uv run python scripts/pylyshyn/generate_samples.py
    uv run python scripts/pylyshyn/generate_samples.py --n-cued 1 3 5 --seeds-per-condition 2
"""

import argparse
import os

from finst_video_model.comprehension.pylyshyn.config import TrialConfig
from finst_video_model.comprehension.pylyshyn.stimulus_gen import generate_stimulus


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-cued", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--seeds-per-condition", type=int, default=1)
    parser.add_argument("--out-root", type=str, default="data/pylyshyn_samples")
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

            ground_truth = generate_stimulus(cfg, trial_dir)

            print(
                f"n_cued={n_cued} probe_on_target={probe_on_target} seed={seed} "
                f"-> {trial_dir} (probed_index={ground_truth['probed_index']}, "
                f"probe_is_target={ground_truth['probe_is_target']})"
            )


if __name__ == "__main__":
    main()
