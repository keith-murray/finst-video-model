"""
Builds the Veo 3.1 text prompt for a trial, given the TrialConfig and the
ground-truth circle count/cue count. The prompt must fully specify the
three-phase structure (cue -> de-cue/track -> re-cue) since Veo only sees a
single static starting frame, not the cueing sequence itself.
"""

from finst_video_model.config import TrialConfig


def build_prompt(cfg: TrialConfig) -> str:
    n = cfg.n_circles
    k = cfg.n_cued

    cue_color_name = "red"
    neutral_color_name = "gray"

    prompt = (
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
    return prompt


if __name__ == "__main__":
    cfg = TrialConfig(n_circles=6, n_cued=2)
    print(build_prompt(cfg))
