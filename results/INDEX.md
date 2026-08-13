# Results index

Batch IDs are opaque hashes assigned by `run_comprehension_batch.py`, so this
file records what each one actually swept. Going forward, new batches should
use a descriptive folder name instead of a hash.

All experiments use the comprehension arm (`scripts/run_comprehension_batch.py`
+ `plot_comprehension_results.py`), n_cued=1 unless noted.

## `99893293` — first real batch (baseline)
- n_circles=4, 1 cued, 20 seeds, model `qwen/qwen3.5-397b-a17b`.
- Near-chance identity accuracy — first evidence the task is hard for VLMs.
- Commit: `4e106cf`.

## `ce675830` — capacity sweep (n_circles)
- n_circles ∈ {2, 3, 4, 6, 8}, model `qwen/qwen3.5-397b-a17b`.
- Headline accuracy-vs-N curve, looking for a Pylyshyn-style capacity knee.
- `position_heuristic_analysis.csv` — retrospective check of whether the
  model just defaults to picking the circle closest to its original (frame 0)
  position rather than actually tracking it.
- Commits: `8dd035e` → `d95ce31` (experiment), `e15e72d` (heuristic analysis).

## `e8489fbd` — capacity x speed sweep (two-stage, same batch_id reused)
- Stage 1: n_circles ∈ {2..8}, speed_px_s fixed at 70. → `accuracy_plot.png`.
- Stage 2: extended in place to a full factorial by adding
  speed_px_s ∈ {70, 87.5, 105, 122.5, 140} for every n_circles value
  (282 rows appended to the same `results.csv`). → `accuracy_plot_speed_px_s.png`.
- Model `qwen/qwen3.5-flash-02-23`, fps=4.
- Commits: `be6352f` (stage 1), `0f31eed` (stage 2 extension).
- Reused (accuracy vs. n_circles panel) in the summary figure below.

## `52aac03d` — fps sweep
- fps ∈ {1, 2, 4, 8, 24}, n_circles=4, speed_px_s=140 fixed.
- Model `qwen/qwen3.5-flash-02-23`.
- Commit: `3eff4d7` (paired with `44f1823e` in the same commit).

## `44f1823e` — speed sweep (companion to fps sweep)
- speed_px_s ∈ {35, 70, 140, 280}, n_circles=4, fps=24 fixed.
- Model `qwen/qwen3.5-flash-02-23`.
- Commit: `3eff4d7` (paired with `52aac03d` in the same commit).

## `ce753d46` — tracking duration x speed sweep
- tracking_s ∈ {4, 6, 8, 10, 12} x speed_px_s ∈ {70, 87.5, 105, 122.5, 140},
  n_circles=5 fixed.
- Model `qwen/qwen3.5-flash-02-23`, fps=4.
- Accuracy decreases monotonically with tracking duration.
- Commit: `0b336de`.
- Reused (accuracy vs. tracking_s panel) in the summary figure below.

## `summary_capacity_and_duration.png` — combined summary figure
- Not a new experiment. `scripts/plot_summary_figure.py` re-plots the
  headline panels from `e8489fbd` (accuracy vs. n_circles) and `ce753d46`
  (accuracy vs. tracking_s) side by side, sharing one speed_px_s color legend.
- Commit: `0d4b989`.
