"""
Generates pylyshyn sample videos (no VLM calls) so the stimulus's timing
and motion speed can be eyeballed directly -- see
claude/2026_08/2026_08_26/TODO.md's "Part 3" and the timing/speed changes
made to finst_video_model.pylyshyn.config on 2026-08-26 at the user's
request ("I wish I had more guidelines as to how fast the stimulus should
be, but let's generate some samples and I can provide feedback").

Sweeps probe_on_target x --speed-scale, where --speed-scale is a multiplier
on TrialConfig's baseline speed_min_px_s/speed_max_px_s (16.0/33.0, itself
already a "slow it down" guess) -- letting several candidate speeds be
eyeballed side by side in one batch rather than guessing a single value
blind. redirect_min_s/redirect_max_s (2.0s) are left at the config default
for every sample; only pixel speed is swept here.

Writes every trial to its own
data/pylyshyn/pylyshyn_samples/<trial_id>/{video.mp4,ground_truth.json}.

Usage:
    uv run python scripts/pylyshyn/generate_samples.py
    uv run python scripts/pylyshyn/generate_samples.py --speed-scale 0.5 1.0 1.5 --seeds-per-condition 2
"""

import argparse
import os

from finst_video_model.pylyshyn.config import TrialConfig
from finst_video_model.pylyshyn.stimulus_gen import generate_stimulus


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speed-scale", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0])
    parser.add_argument("--probe-on-target", type=int, nargs="+", default=[1, 0])
    parser.add_argument("--seeds-per-condition", type=int, default=1)
    parser.add_argument("--out-root", type=str, default="data/pylyshyn/pylyshyn_samples")
    args = parser.parse_args()

    os.makedirs(args.out_root, exist_ok=True)

    base = TrialConfig()
    conditions = [
        (scale, bool(probe_on_target))
        for scale in args.speed_scale
        for probe_on_target in args.probe_on_target
    ]

    for scale, probe_on_target in conditions:
        for rep in range(args.seeds_per_condition):
            seed = int(scale * 1000) + (0 if probe_on_target else 500) + rep
            cfg = TrialConfig(
                probe_on_target=probe_on_target,
                speed_min_px_s=base.speed_min_px_s * scale,
                speed_max_px_s=base.speed_max_px_s * scale,
                seed=seed,
            )
            trial_dir = os.path.join(args.out_root, cfg.trial_id)
            os.makedirs(trial_dir, exist_ok=True)

            ground_truth, _frames = generate_stimulus(cfg, trial_dir)

            print(
                f"speed_scale={scale} probe_on_target={probe_on_target} seed={seed} "
                f"speed_px_s=[{cfg.speed_min_px_s:.1f},{cfg.speed_max_px_s:.1f}] -> {trial_dir} "
                f"(probe_is_target={ground_truth['probe_is_target']})"
            )


if __name__ == "__main__":
    main()
