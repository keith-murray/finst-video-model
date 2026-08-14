"""
Renders the comprehension arm's base video with no label phase (cue -> track
only, motion never freezes or gets labeled) and builds the prompt that would
be sent to a video *extension* model -- one that continues an existing video
from its last frame, e.g. via OpenRouter's video-extend feature -- asking it
to recolor the originally-cued circles in the extended footage.

This is the "video-extension model" option floated in
claude/2026_07_20/TODO.md as an alternative to the comprehension (VLM
text-question) arm: instead of asking a model a question in text, we ask a
video-extend model to perform the re-cue itself, closer to the original Veo
generation-arm task. No model call happens here yet -- OpenRouter video-extend
support for any specific model hasn't been wired up, and few such models
exist outside beta -- this script only produces the base video + extend
prompt so that integration can be added later without touching stimulus
generation.

The prompt lives here (not in the shared package), same convention as
scripts/run_trial.py, scripts/run_circular_trial.py, and
scripts/run_comprehension_trial.py.

Usage:
    uv run python scripts/run_extend_trial.py
"""

import json
import os

from finst_video_model.comprehension.config import TrialConfig
from finst_video_model.comprehension.stimulus_gen import generate_stimulus

EXTEND_S = 1.0  # requested duration of the extended (re-cue) footage


def build_extend_prompt(cfg: TrialConfig, extend_s: float) -> str:
    k = cfg.n_cued
    plural = "circle" if k == 1 else "circles"

    return (
        f"Continue this video for {extend_s:.1f} more seconds. All circles "
        f"are currently plain gray and moving; bring them to a complete "
        f"stop within the first half-second of the new footage and do not "
        f"let them move again for the rest of the video. Once stopped, "
        f"recolor only the {plural} that were red at the very start of the "
        f"original video -- before they turned gray -- back to red; every "
        f"other circle stays gray. Exactly {k} circle(s) should be red by "
        f"the end of the extended footage, no more and no fewer. Do not "
        f"change the number of circles, their sizes, or the background at "
        f"any point."
    )


def main():
    cfg = TrialConfig(
        n_circles=6, n_cued=2, seed=42, use_label_phase=False,
        cue_flash_s=1.0, tracking_s=3.0,  # 4s base video total
    )
    trial_dir = os.path.join("data", cfg.trial_id)
    os.makedirs(trial_dir, exist_ok=True)

    ground_truth = generate_stimulus(cfg, trial_dir)
    prompt = build_extend_prompt(cfg, EXTEND_S)

    with open(os.path.join(trial_dir, "extend_prompt.txt"), "w") as f:
        f.write(prompt)

    result = {
        "trial_id": cfg.trial_id,
        "video_path": ground_truth["video_path"],
        "cued_indices": ground_truth["cued_indices"],
        "extend_prompt": prompt,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
