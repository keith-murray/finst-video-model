"""
Runs one pylyshyn-arm trial end to end: renders a "cue -> track -> probe"
video ourselves, builds the True/False question text, sends both to a
video-understanding VLM via OpenRouter chat completions, and scores the
response (hit/miss/false_alarm/correct_rejection -- see
finst_video_model.scoring.classify_trial) against ground
truth.

The prompt/question and model choice (MODEL) live here (not in the shared
package), same convention as scripts/prototype/run_comprehension_trial.py.

Usage:
    uv run python scripts/pylyshyn/run_pylyshyn_trial.py
    uv run python scripts/pylyshyn/run_pylyshyn_trial.py --seed 7 --n-cued 1 --no-probe-on-target
"""

import argparse
import json
import os

import requests

from finst_video_model.pylyshyn.config import TrialConfig
from finst_video_model.pylyshyn.stimulus_gen import generate_stimulus
from finst_video_model.scoring import classify_trial, parse_boolean_answer
from finst_video_model.vlm_client import ask_about_video

MODEL = "google/gemini-2.5-flash"


def build_question(cfg: TrialConfig) -> str:
    k = cfg.n_cued
    plural = "cross" if k == 1 else "crosses"
    verb = "is" if k == 1 else "are"
    pronoun = "it turns" if k == 1 else "they turn"

    return (
        f"You will watch a video of {cfg.n_objects} white crosses (+) on a "
        f"black background. At the start of the video, all crosses are "
        f"stationary, and {k} of them -- the cued {plural} -- {verb} colored "
        f"red, while the rest stay white. After a moment {pronoun} white "
        f"too, and every cross begins moving continuously, independently, "
        f"and unpredictably -- there is no more color difference between "
        f"crosses once they start moving, so you can only keep track of "
        f"which cross is which by following its motion. At some point while "
        f"the crosses are moving, exactly one of them briefly turns red for "
        f"about two seconds, then turns back to white, and all crosses keep "
        f"moving until the video ends.\n\n"
        f"Question: was the cross that briefly turned red "
        + ("the cross that was red" if k == 1 else "one of the crosses that were red")
        + f" at the very start of the video?\n\n"
        f"Answer with only the single word True or False. Do not include "
        f"any other text in your answer."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default=MODEL)
    parser.add_argument("--n-cued", type=int, default=3)
    parser.add_argument(
        "--probe-on-target", action=argparse.BooleanOptionalAction, default=True,
        help="Whether the probe should land on a cued target (hit/miss trial) "
             "or a distractor (false-alarm/correct-rejection trial)",
    )
    args = parser.parse_args()

    cfg = TrialConfig(n_cued=args.n_cued, probe_on_target=args.probe_on_target, seed=args.seed)
    trial_dir = os.path.join("data", cfg.trial_id)
    os.makedirs(trial_dir, exist_ok=True)

    ground_truth, _frames = generate_stimulus(cfg, trial_dir)
    question = build_question(cfg)
    with open(os.path.join(trial_dir, "question.txt"), "w") as f:
        f.write(question)

    result = {
        "trial_id": cfg.trial_id,
        "model": args.model,
        "seed": args.seed,
        "n_cued": args.n_cued,
        "video_path": ground_truth["video_path"],
        "question": question,
        "probe_is_target": ground_truth["probe_is_target"],
    }

    try:
        response_text = ask_about_video(ground_truth["video_path"], question, args.model)
        predicted = parse_boolean_answer(response_text)
        result["response"] = response_text
        result["predicted"] = predicted
        result["outcome"] = (
            classify_trial(ground_truth["probe_is_target"], predicted)
            if predicted is not None else "unparseable"
        )
    except (requests.RequestException, KeyError, RuntimeError) as e:
        result["error"] = str(e)

    with open(os.path.join(trial_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
