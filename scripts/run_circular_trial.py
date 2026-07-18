"""
Runs one real Veo 3.1 "circular track" trial via OpenRouter.

Unlike the free-form bouncing-circles trial (see run_trial.py), the motion
here is much more constrained: circles start evenly spaced around a large
circular track drawn in the frame, and are instructed to move clockwise
along that track rather than bounce around freely. This is a deliberate
simplification (per claude/2026_07_18/TODO.md) to reduce the amount of
hallucinated motion Veo has room to introduce.

The prompt lives here (not in the shared package) so each experiment's
wording can be iterated on independently.

Usage:
    uv run python scripts/run_circular_trial.py
"""

import json
import os

from finst_video_model.config import TrialConfig
from finst_video_model.stimulus_gen import generate_circular_track_stimulus
from finst_video_model.veo_client import run_trial


def build_prompt(cfg: TrialConfig) -> str:
    n = cfg.n_circles
    k = cfg.n_cued

    cue_color_name = "red"
    neutral_color_name = "gray"

    return (
        f"A static wide shot of {n} identical plain {neutral_color_name} and "
        f"{cue_color_name} circles arranged evenly spaced around a large "
        f"circular track drawn as a thin gray line on a plain light "
        f"background, filmed from directly above, camera fixed and unmoving "
        f"for the entire video. "
        f"Exactly {k} of the circles start {cue_color_name}; the remaining "
        f"{n - k} are {neutral_color_name}. "
        f"Within the first {cfg.cue_flash_s:.1f} second(s), all circles "
        f"become identical plain {neutral_color_name} -- no circle is "
        f"{cue_color_name} anymore, and no circle has any label, number, or "
        f"marking of any kind. "
        f"For the next {cfg.tracking_s:.1f} seconds, all circles move "
        f"clockwise along the circular track at the same constant speed, "
        f"staying exactly on the track at all times, maintaining their "
        f"original order and even spacing around the track, never leaving "
        f"the track, merging, splitting, passing through each other, "
        f"appearing, disappearing, or changing size. The circular track "
        f"itself remains visible and unchanged throughout. "
        f"After the {cfg.tracking_s:.1f} seconds of movement, all circles "
        f"come to a complete stop and do not move again for the rest of the "
        f"video. "
        f"In the final {cfg.recue_s:.1f} second(s) of the video, while the "
        f"circles remain stopped, the circle(s) that were {cue_color_name} "
        f"at the very start of the video -- and only those circles -- "
        f"become {cue_color_name} again; every other circle remains "
        f"{neutral_color_name}. "
        f"Exactly {k} circle(s) should be {cue_color_name} in the last "
        f"frame, no more and no fewer. No text, numbers, or labels appear "
        f"anywhere in the video."
    )


def main():
    cfg = TrialConfig(n_circles=2, n_cued=1, seed=42)
    trial_dir = os.path.join("data", cfg.trial_id)
    os.makedirs(trial_dir, exist_ok=True)

    ground_truth = generate_circular_track_stimulus(cfg, trial_dir)
    prompt = build_prompt(cfg)

    result = run_trial(cfg, trial_dir, prompt, ground_truth["frame0_path"])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
