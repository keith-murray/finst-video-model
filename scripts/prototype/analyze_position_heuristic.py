"""
Retrospective test of an alternative hypothesis for the comprehension arm's
capacity-curve result: rather than actually tracking the cued circle(s)
through the de-cued phase, the model might just answer with whichever
letter ends up spatially nearest to the cued circle's *original* (cue-phase)
position -- a "nearest-to-start" heuristic that would produce a capacity-like
accuracy decline for the wrong reason (more circles -> more label positions
crowded near the original spot -> heuristic gets noisier), without any real
identity tracking happening at all.

For every already-run trial in a run_comprehension_batch.py batch, this
recomputes each circle's *start* position by re-deriving it from the
trial's saved config (build_circles is a pure, deterministic function of
seed/n_circles/etc, so this needs no new video/API calls -- the batch's
existing data/<batch_id>/trials/<trial_id>/ground_truth.json already has
each circle's *final* position and label). For each cued circle, finds
whichever label is nearest to that circle's start position (the heuristic's
predicted letter), then compares:
  - whether the heuristic's letter(s) equal the true cued letter(s)
    ("heuristic_correct" -- how often would blind position-matching alone
    get this trial right by chance)
  - whether the model's actual answer matches the heuristic vs. the truth,
    split by whether the trial was a "conflict" (heuristic != truth) or
    "agreement" (heuristic == truth) trial. Conflict trials are the
    informative ones: if the model is real-tracking, its accuracy there
    should look like its overall accuracy; if it's using the heuristic,
    accuracy on conflict trials should collapse while model-matches-heuristic
    stays high.

Writes results/<batch_id>/position_heuristic_analysis.csv (one row per
scored trial) and prints the conflict/agreement breakdown.

Usage:
    uv run python scripts/analyze_position_heuristic.py results/<batch_id>
"""

import argparse
import csv
import json
import os

from finst_video_model.comprehension.config import TrialConfig
from finst_video_model.comprehension.physics import build_circles

OUT_FIELDS = [
    "trial_id", "n_circles", "seed", "true_letters", "heuristic_letters",
    "predicted_letters", "heuristic_correct", "model_matches_truth",
    "model_matches_heuristic",
]


def nearest_label(x: float, y: float, circles_meta: list[dict]) -> str:
    best = min(circles_meta, key=lambda c: (c["final_x"] - x) ** 2 + (c["final_y"] - y) ** 2)
    return best["label"]


def analyze_trial(trial_dir: str) -> dict | None:
    gt_path = os.path.join(trial_dir, "ground_truth.json")
    if not os.path.isfile(gt_path):
        return None
    with open(gt_path) as f:
        gt = json.load(f)

    cfg = TrialConfig(**gt["config"])
    circles, cued_indices = build_circles(cfg)  # deterministic: re-derives start positions
    start_by_index = {c.index: (c.x, c.y) for c in circles}

    heuristic_letters = sorted(
        nearest_label(*start_by_index[i], gt["circles"]) for i in cued_indices
    )
    true_letters = gt["cued_letters"]
    return {"heuristic_letters": heuristic_letters, "true_letters": true_letters}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "batch_dir",
        help="Path to a results/<batch_id> directory produced by run_comprehension_batch.py",
    )
    args = parser.parse_args()

    batch_id = os.path.basename(os.path.normpath(args.batch_dir))
    trials_dir = os.path.join("data", batch_id, "trials")
    results_csv = os.path.join(args.batch_dir, "results.csv")

    with open(results_csv) as f:
        rows = list(csv.DictReader(f))

    out_rows = []
    for row in rows:
        if row["error"]:
            continue  # no model answer to compare against
        trial = analyze_trial(os.path.join(trials_dir, row["trial_id"]))
        if trial is None:
            continue

        predicted_letters = sorted(row["predicted_letters"].split(",")) if row["predicted_letters"] else []
        heuristic_correct = trial["heuristic_letters"] == trial["true_letters"]
        model_matches_truth = predicted_letters == trial["true_letters"]
        model_matches_heuristic = predicted_letters == trial["heuristic_letters"]

        out_rows.append({
            "trial_id": row["trial_id"],
            "n_circles": row["n_circles"],
            "seed": row["seed"],
            "true_letters": ",".join(trial["true_letters"]),
            "heuristic_letters": ",".join(trial["heuristic_letters"]),
            "predicted_letters": ",".join(predicted_letters),
            "heuristic_correct": heuristic_correct,
            "model_matches_truth": model_matches_truth,
            "model_matches_heuristic": model_matches_heuristic,
        })

    out_path = os.path.join(args.batch_dir, "position_heuristic_analysis.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Wrote {out_path} ({len(out_rows)} scored trials)")

    conflict = [r for r in out_rows if not r["heuristic_correct"]]
    agreement = [r for r in out_rows if r["heuristic_correct"]]

    def rate(rows, key):
        return sum(r[key] for r in rows) / len(rows) if rows else float("nan")

    print(f"\nOverall: {len(out_rows)} trials, {len(agreement)} agreement "
          f"(heuristic == truth), {len(conflict)} conflict (heuristic != truth)")
    print(f"Overall model accuracy (matches truth): {rate(out_rows, 'model_matches_truth'):.2f}")
    print(f"Overall model matches heuristic:        {rate(out_rows, 'model_matches_heuristic'):.2f}")
    print()
    print("Agreement trials (heuristic can't be distinguished from truth here):")
    print(f"  model accuracy:          {rate(agreement, 'model_matches_truth'):.2f}  (n={len(agreement)})")
    print()
    print("Conflict trials (heuristic and truth disagree -- the informative subset):")
    print(f"  model matches TRUTH:     {rate(conflict, 'model_matches_truth'):.2f}  (n={len(conflict)})")
    print(f"  model matches HEURISTIC: {rate(conflict, 'model_matches_heuristic'):.2f}  (n={len(conflict)})")
    print()
    print("If accuracy on conflict trials collapses relative to agreement trials while")
    print("model-matches-heuristic stays high, that supports the position heuristic.")
    print("If conflict-trial accuracy looks like overall accuracy, that argues against it.")


if __name__ == "__main__":
    main()
