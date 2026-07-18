"""
Runs one real Veo 3.1 "bouncing circles" trial via OpenRouter: generates the
frame-0 stimulus image (circles scattered randomly, free-form physics), builds
the trial's prompt, submits the job, polls to completion, and downloads the
resulting video.

The prompt lives here (not in the shared package) so each experiment's
wording can be iterated on independently.

Usage:
    uv run python scripts/run_trial.py
"""

import json
import os

from finst_video_model.config import TrialConfig
from finst_video_model.stimulus_gen import generate_stimulus
from finst_video_model.veo_client import run_trial


def build_prompt(cfg: TrialConfig) -> str:
    n = cfg.n_circles
    k = cfg.n_cued

    cue_color_name = "red"
    neutral_color_name = "gray"

    return (
        f"A static wide shot of {n} identical plain {neutral_color_name} and "
        f"{cue_color_name} circles on a plain light background, filmed from "
        f"directly above, camera fixed and unmoving for the entire video. "
        f"Exactly {k} of the circles start {cue_color_name}; the remaining "
        f"{n - k} are {neutral_color_name}. "
        f"Within the first {cfg.cue_flash_s:.1f} second(s), all circles "
        f"become identical plain {neutral_color_name} -- no circle is "
        f"{cue_color_name} anymore, and no circle has any label, number, or "
        f"marking of any kind. "
        f"For the next {cfg.tracking_s:.1f} seconds, all circles are plain "
        f"{neutral_color_name} and indistinguishable from one another in "
        f"appearance, {cfg.physics_description}. Circles do not merge, "
        f"split, appear, disappear, or change size at any point. "
        f"In the final {cfg.recue_s:.1f} second(s) of the video, the circles "
        f"that were {cue_color_name} at the very start of the video -- and "
        f"only those circles -- become {cue_color_name} again; every other "
        f"circle remains {neutral_color_name}. "
        f"Exactly {k} circles should be {cue_color_name} in the last frame, "
        f"no more and no fewer. No text, numbers, or labels appear anywhere "
        f"in the video."
    )


def main():
    cfg = TrialConfig(n_circles=6, n_cued=2, seed=42)
    trial_dir = os.path.join("data", cfg.trial_id)
    os.makedirs(trial_dir, exist_ok=True)

    ground_truth = generate_stimulus(cfg, trial_dir)
    prompt = build_prompt(cfg)

    result = run_trial(cfg, trial_dir, prompt, ground_truth["frame0_path"])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
