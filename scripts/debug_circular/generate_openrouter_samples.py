"""
Generates debug-circular sample videos with an artificially STRETCHED mp4
duration, to test whether an OpenRouter-hosted model's server-side video
frame sampling is duration/fps-driven -- see
claude/2026_08/2026_08_26/TODO.md's "Part 2: Dealing with unreasonably long
wait times".

Each trial renders the exact same content as scripts/debug_circular/generate_samples.py
(same TrialConfig, same native fps=10 physics -- via the unmodified
generate_stimulus()), but the mp4 written here is re-encoded at a much
lower --encode-fps (default 2.0) via the new
finst_video_model.debug_circular.stimulus_gen.write_mp4() helper. Same 100
frames either way -- only the mp4 container's declared fps/duration changes
(100 frames / 10fps = 10s native vs. 100 frames / 2fps = 50s stretched).
The hypothesis: OpenRouter's downsampler may retain more of an already-fixed
frame budget from a video that presents as long-and-slow rather than
short-and-fast.

Kept as a separate script (rather than adding an --encode-fps flag to
generate_samples.py) and a separate output directory (rather than
data/debug_circular/debug_circular_samples/) per the user's request, so the
existing cluster-oriented sample generator and its output stay untouched.

Usage:
    uv run python scripts/debug_circular/generate_openrouter_samples.py
    uv run python scripts/debug_circular/generate_openrouter_samples.py --rotation-deg 45 180 --encode-fps 1
"""

import argparse
import os

import numpy as np

from finst_video_model.debug_circular.config import TrialConfig
from finst_video_model.debug_circular.stimulus_gen import generate_stimulus, write_mp4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rotation-deg", type=float, nargs="+", default=[45, 90, 180, 360])
    parser.add_argument("--n-objects", type=int, default=3)
    parser.add_argument("--clockwise", type=int, default=[True, False])
    parser.add_argument("--probe-on-target", type=int, default=[True, False])
    parser.add_argument("--seeds-per-condition", type=int, default=1)
    parser.add_argument(
        "--encode-fps", type=float, default=2.0,
        help="fps the stretched video.mp4 is encoded at -- decoupled from the "
             "native fps=10 physics/frame content. Default 2.0 stretches the "
             "native 10s trial to a nominal 50s.",
    )
    parser.add_argument(
        "--out-root", type=str, default="data/debug_circular/openrouter_stretch_samples",
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

            ground_truth = generate_stimulus(cfg, trial_dir, save_mp4=False)

            frames = np.load(f"{trial_dir}/video.npy")
            write_mp4(frames, f"{trial_dir}/video.mp4", fps=args.encode_fps)

            native_duration_s = frames.shape[0] / cfg.fps
            stretched_duration_s = frames.shape[0] / args.encode_fps
            print(
                f"rotation_deg={rotation_deg} clockwise={clockwise} "
                f"probe_on_target={probe_on_target} seed={seed} -> {trial_dir} "
                f"({frames.shape[0]} frames, native {native_duration_s:.1f}s @ "
                f"{cfg.fps}fps -> stretched {stretched_duration_s:.1f}s @ "
                f"{args.encode_fps}fps, probe_is_target={ground_truth['probe_is_target']})"
            )


if __name__ == "__main__":
    main()
