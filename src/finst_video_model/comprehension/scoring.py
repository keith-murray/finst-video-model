"""
Parses a VLM's free-text answer into a set of letters and scores it against
ground truth. Kept separate from vlm_client.py since this parsing/scoring
logic is reusable across whichever model or question wording a script
chooses to use.
"""

import re


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
