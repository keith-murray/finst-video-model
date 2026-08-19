"""
Builds the text question sent alongside the video to the VLM. Unlike the
Veo arm, the model only needs to answer in text -- it never has to generate
correct motion itself, so the prompt can be a direct instruction rather
than a scene description.
"""

from config import TrialConfig


def build_question(cfg: TrialConfig) -> str:
    k = cfg.n_cued
    plural = "circle" if k == 1 else "circles"

    question = (
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
    return question


if __name__ == "__main__":
    cfg = TrialConfig(n_circles=6, n_cued=2)
    print(build_question(cfg))
