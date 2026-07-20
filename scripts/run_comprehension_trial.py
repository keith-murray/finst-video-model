"""
Runs one comprehension-arm trial end to end: renders a "cue -> track ->
label" video ourselves, builds the text question, sends both to a
video-understanding VLM via OpenRouter chat completions, and scores the
response against ground truth.

Unlike the Veo arm, no video-generation call is involved -- the model only
has to answer in text, via finst_video_model.comprehension.vlm_client.

The prompt/question and model choice (MODEL) live here (not in the shared
package) so each experiment's wording and model can be iterated on
independently, same convention as scripts/run_trial.py and
scripts/run_circular_trial.py.

Usage:
    uv run python scripts/run_comprehension_trial.py
    uv run python scripts/run_comprehension_trial.py --seed 7 --model qwen/qwen3.5-397b-a17b
"""

import argparse
import json
import os

import requests

from finst_video_model.comprehension.config import TrialConfig
from finst_video_model.comprehension.scoring import parse_answer_letters, score_answer
from finst_video_model.comprehension.stimulus_gen import generate_stimulus
from finst_video_model.comprehension.vlm_client import ask_about_video

MODEL = "google/gemini-2.5-flash"


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default=MODEL)
    parser.add_argument("--n-circles", type=int, default=6)
    parser.add_argument("--n-cued", type=int, default=2)
    args = parser.parse_args()

    cfg = TrialConfig(n_circles=args.n_circles, n_cued=args.n_cued, seed=args.seed)
    trial_dir = os.path.join("data", cfg.trial_id)
    os.makedirs(trial_dir, exist_ok=True)

    ground_truth = generate_stimulus(cfg, trial_dir)

    result = {
        "trial_id": cfg.trial_id,
        "model": args.model,
        "seed": args.seed,
        "video_path": ground_truth["video_path"],
        "cued_letters": ground_truth["cued_letters"],
    }

    if cfg.use_label_phase:
        question = build_question(cfg)
        result["question"] = question
        with open(os.path.join(trial_dir, "question.txt"), "w") as f:
            f.write(question)

        try:
            response_text = ask_about_video(ground_truth["video_path"], question, args.model)
            result["response"] = response_text
            predicted_letters = parse_answer_letters(response_text)
            result["score"] = score_answer(predicted_letters, ground_truth["cued_letters"])
        except (requests.RequestException, KeyError, RuntimeError) as e:
            result["error"] = str(e)

    with open(os.path.join(trial_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
