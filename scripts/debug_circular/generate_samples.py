"""
Generates a handful of sample debug-circular videos so the stimulus (a
rigid ring of crosses, one cue/probe flash, one rotation) can be eyeballed
before any VLM is wired up. See `claude/2026_08/2026_08_25/TODO.md` for the
full design rationale.

Sweeps rotation_deg (the "variety of speeds" the TODO calls for) and
probe_on_target, and crosses every condition with both rotation directions
so clockwise/counter-clockwise trials always come out equal in count.
Writes every trial to its own `data/debug_circular_samples/<trial_id>/`
directory, with both `video.mp4` (for eyeballing) and `video.npy` (raw RGB
frames -- what the locally-hosted Qwen3.8-27B actually receives).

Usage:
    uv run python scripts/debug_circular/generate_samples.py
    uv run python scripts/debug_circular/generate_samples.py --rotation-deg 45 180 --seeds-per-condition 2
"""

import argparse
import os

from finst_video_model.debug_circular.config import TrialConfig
from finst_video_model.debug_circular.stimulus_gen import generate_stimulus


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rotation-deg", type=float, nargs="+", default=[45, 90, 180, 360])
    parser.add_argument("--n-objects", type=int, default=3)
    parser.add_argument("--clockwise", type=int, default=[True,False])
    parser.add_argument("--probe-on-target", type=int, default=[True,False])
    parser.add_argument("--seeds-per-condition", type=int, default=1)
    parser.add_argument("--out-root", type=str, default="data/debug_circular/debug_circular_samples")
    parser.add_argument(
        "--save-mp4", action=argparse.BooleanOptionalAction, default=True,
        help="Also render video.mp4 alongside video.npy. Pass --no-save-mp4 "
             "when generating directly on the cluster, whose compute nodes "
             "have no GPU video-encode device and can't reliably produce mp4s.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_root, exist_ok=True)

    conditions = [
        (rotation_deg, probe_on_target, clockwise)
        for rotation_deg in args.rotation_deg
        for probe_on_target in args.probe_on_target
        for clockwise in args.clockwise
    ]

    for rotation_deg, probe_on_target, clockwise in conditions:
        for rep in range(args.seeds_per_condition):
            seed = (
                int(rotation_deg * 10)
                + (0 if probe_on_target else 5000)
                + (0 if clockwise else 100000)
                + rep
            )
            cfg = TrialConfig(
                n_objects=args.n_objects,
                rotation_deg=rotation_deg,
                clockwise=clockwise,
                probe_on_target=probe_on_target,
                seed=seed,
            )
            trial_dir = os.path.join(args.out_root, cfg.trial_id)
            os.makedirs(trial_dir, exist_ok=True)

            ground_truth = generate_stimulus(cfg, trial_dir, save_mp4=args.save_mp4)

            print(
                f"rotation_deg={rotation_deg} clockwise={clockwise} "
                f"probe_on_target={probe_on_target} seed={seed} -> {trial_dir} "
                f"(probed_index={ground_truth['probed_index']}, "
                f"probe_is_target={ground_truth['probe_is_target']})"
            )


if __name__ == "__main__":
    main()
