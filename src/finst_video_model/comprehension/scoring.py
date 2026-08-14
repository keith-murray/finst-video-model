"""
Parses a VLM's free-text answer and scores it against ground truth. Kept
separate from vlm_client.py since this parsing/scoring logic is reusable
across whichever model or question wording a script chooses to use.

`parse_answer_letters`/`score_answer` are for the old end-of-clip
letter-labeling report and are currently unused by either siloed
comprehension arm (`pylyshyn`, `smooth_pursuit`) -- both now instead use a
single True/False probe report, scored via `parse_boolean_answer` /
`classify_trial` / `compute_d_prime` below. Shared here (not siloed into
either arm) because both arms produce the identical `probe_is_target`
ground-truth shape.
"""

import re

from scipy.stats import norm


def parse_answer_letters(response_text: str) -> set[str]:
    """Extracts standalone capital letters (A-Z) from a free-text VLM
    response, e.g. "A, C" or "The answer is A and C" -> {"A", "C"}. Only
    matches single-letter tokens (bounded by non-word characters), so
    letters embedded in ordinary words (e.g. the "A" in "AVERAGE") are not
    picked up."""
    return set(re.findall(r"\b[A-Z]\b", response_text.upper()))


def score_answer(predicted_letters: set[str], cued_letters: list[str]) -> dict:
    """Compares a parsed answer against ground_truth["cued_letters"].
    `exact_match` is the strict identity-correctness signal; `correct_count`
    is the "cheap" success mode (right number of letters, possibly wrong
    identity); precision/recall give partial credit when the model gets
    some but not all letters right."""
    cued = set(cued_letters)

    true_positives = predicted_letters & cued
    precision = len(true_positives) / len(predicted_letters) if predicted_letters else 0.0
    recall = len(true_positives) / len(cued) if cued else 0.0

    return {
        "predicted_letters": sorted(predicted_letters),
        "cued_letters": sorted(cued),
        "exact_match": predicted_letters == cued,
        "correct_count": len(predicted_letters) == len(cued),
        "precision": precision,
        "recall": recall,
    }


def parse_boolean_answer(response_text: str) -> bool | None:
    """Extracts a True/False judgment from a free-text VLM response, e.g.
    "True" or "The answer is False." -> True/False. Matches whole words
    only (so "truely" doesn't match). Returns None if both or neither word
    appears, rather than guessing -- callers must treat None as an
    unscorable trial, not as a False."""
    text = response_text.lower()
    has_true = re.search(r"\btrue\b", text) is not None
    has_false = re.search(r"\bfalse\b", text) is not None
    if has_true == has_false:
        return None
    return has_true


def classify_trial(probe_is_target: bool, predicted: bool) -> str:
    """Classifies one trial's outcome against the 2x2 signal-detection
    table (probe_is_target x predicted): "hit", "miss", "false_alarm", or
    "correct_rejection"."""
    if probe_is_target:
        return "hit" if predicted else "miss"
    return "false_alarm" if predicted else "correct_rejection"


def compute_d_prime(trials: list[dict]) -> dict:
    """Computes d' (and the companion bias metric c) from a list of trials,
    each a dict with "probe_is_target" (bool) and "predicted" (bool or None
    -- None marks an unparseable answer, excluded from the counts below).

    d' can't be computed from a single trial -- it requires pooling many
    probe_is_target=True trials into a hit rate and many
    probe_is_target=False trials into a false-alarm rate, then comparing
    them via the inverse normal CDF (probit). Applies the Hautus (1995)
    log-linear correction unconditionally (not just at 0%/100% rates, so
    the correction doesn't itself introduce a discontinuity at the
    boundary) to keep the rates away from exactly 0 or 1, where the probit
    is +-inf.
    """
    hits = misses = false_alarms = correct_rejections = n_unparseable = 0
    for trial in trials:
        predicted = trial["predicted"]
        if predicted is None:
            n_unparseable += 1
            continue
        outcome = classify_trial(trial["probe_is_target"], predicted)
        if outcome == "hit":
            hits += 1
        elif outcome == "miss":
            misses += 1
        elif outcome == "false_alarm":
            false_alarms += 1
        else:
            correct_rejections += 1

    n_target_trials = hits + misses
    n_distractor_trials = false_alarms + correct_rejections

    hit_rate = (hits + 0.5) / (n_target_trials + 1)
    false_alarm_rate = (false_alarms + 0.5) / (n_distractor_trials + 1)

    z_hit = norm.ppf(hit_rate)
    z_fa = norm.ppf(false_alarm_rate)

    return {
        "n_target_trials": n_target_trials,
        "n_distractor_trials": n_distractor_trials,
        "n_unparseable": n_unparseable,
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
        "correct_rejections": correct_rejections,
        "hit_rate": hit_rate,
        "false_alarm_rate": false_alarm_rate,
        "d_prime": z_hit - z_fa,
        "criterion": -0.5 * (z_hit + z_fa),
    }
