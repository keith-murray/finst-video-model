"""
Runs one comprehension-arm trial: renders a "cue -> track -> label" video
ourselves (no OpenRouter video-generation call involved) and builds the text
question that would be sent to a video-understanding VLM alongside it.

Unlike the Veo arm, this script does not yet call any model -- the VLM API
wrapper, answer parsing, and scoring are still unbuilt (see README.md). This
script only exercises stimulus generation end to end and writes the question
next to the video/ground truth so it can be inspected or fed to a VLM by
hand.

The prompt/question lives here (not in the shared package) so each
experiment's wording can be iterated on independently, same convention as
scripts/run_trial.py and scripts/run_circular_trial.py.

Usage:
    uv run python scripts/run_comprehension_trial.py
"""

import json
import os

from finst_video_model.comprehension.config import TrialConfig
from finst_video_model.comprehension.stimulus_gen import generate_stimulus


def build_question(cfg: TrialConfig) -> str:
    if not cfg.use_label_phase:
        raise ValueError(
            "build_question requires cfg.use_label_phase=True -- without "
            "the label phase there are no letters for the model to report."
        )

    k = cfg.n_cued
    plural = "circle" if k == 1 else "circles"

    return (
        f"You will watch a video of identical circles moving on a plain "
        f"background. At the very start of the video, {k} of the circles "
        f"are briefly colored red; all other circles are gray. Within about "
        f"a second, the red circles turn gray as well, so for most of the "
        f"video every circle is plain gray and visually indistinguishable "
        f"from the others -- you can only keep track of which circle is "
        f"which by following its motion. Near the end of the video, all "
        f"circles stop moving and each one is labeled with a single letter.\n\n"
        f"Question: which letter(s) label the {plural} that were red at the "
        f"very start of the video?\n\n"
        f"Answer with only the letter(s), separated by commas if there is "
        f"more than one (for example: \"A, C\"). Do not include any other "
        f"text in your answer."
    )


def main():
    cfg = TrialConfig(n_circles=6, n_cued=2, seed=42)
    trial_dir = os.path.join("data", cfg.trial_id)
    os.makedirs(trial_dir, exist_ok=True)

    ground_truth = generate_stimulus(cfg, trial_dir)

    result = {"trial_id": cfg.trial_id, "video_path": ground_truth["video_path"]}
    if cfg.use_label_phase:
        question = build_question(cfg)
        result["question"] = question
        with open(os.path.join(trial_dir, "question.txt"), "w") as f:
            f.write(question)

    result["cued_letters"] = ground_truth["cued_letters"]
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
