# Session summary — 2026-07-22

Added a video-length (tracking-duration) sweep axis to the comprehension
arm's batch runner, ran it, backfilled a handful of errored trials, and
built a two-panel summary figure combining today's result with yesterday's
n_circles x speed grid for the user's PI.

## Batch runner: tracking_s as a sweepable axis

`scripts/run_comprehension_batch.py` previously swept `n_circles`, `fps`,
and `speed_px_s` as list args, but `cue_flash_s`/`tracking_s`/`label_s` were
fixed scalars per run. `tracking_s` is now a list-sweepable axis exactly
like `fps`/`speed_px_s` (`--tracking-s` takes `nargs="+"`, is part of the
resume/`already_ran` key, and is saved in `config.json`), so a run can hold
`cue_flash_s`/`label_s` fixed while varying tracking duration -- i.e. video
length -- as its own grid dimension. Each result row also gets a derived
`total_duration_s = cue_flash_s + tracking_s + label_s` column, so a
tracking_s sweep can be read/plotted directly in terms of overall video
length without mental arithmetic. `plot_comprehension_results.py` needed no
changes -- its `--x-axis`/`--series-by` flags already read any column
generically.

## Video-length sweep (`results/ce753d46/`)

The user ran the actual sweep by hand (parameters diverged slightly from
what was proposed) -- `n_circles=5, n_cued=1`, `fps=4`,
`speed_px_s=[70, 87.5, 105, 122.5, 140]` (same five speeds as yesterday's
`e8489fbd` grid), `tracking_s=[4, 6, 8, 10, 12]`, `cue_flash_s=2`,
`label_s=2`, 10 seeds/condition, `qwen/qwen3.5-flash-02-23` -- 250 trials
total.

6/250 trials errored on the first pass (1 JSON-parse failure, 5 "response
ended prematurely" -- both familiar intermittent OpenRouter failure modes
from prior sessions, not correlated with any particular condition).
Backfilled by stripping the 6 error rows from `results.csv` and
`--resume ce753d46`; all 250 scored cleanly afterward.

**Result**: a clean, monotonic decline in accuracy as tracking duration
increases, holding across all five speeds -- roughly 50-70% accuracy at
`tracking_s=4` down to 0-20% at `tracking_s=12`. Speed=70 (the slowest)
holds up best at the longer durations, consistent with the prior
speed-rescue finding, but the ordering among the four faster speeds is noisy
and overlapping at n=10/condition -- no clean monotonic speed ranking within
this range, just the general "slower survives longer" trend at the
extremes. Plot at `results/ce753d46/accuracy_plot_tracking_s_speed_px_s.png`.

## Two-panel PI summary figure

New `scripts/plot_summary_figure.py` combines the "exact identity match"
panel from two sweeps side by side: left is accuracy vs. `n_circles` from
yesterday's `results/e8489fbd/` grid, right is accuracy vs. `tracking_s`
from today's `results/ce753d46/` sweep. Both used the same five
`speed_px_s` values, so the two panels share one color-coded legend across
the top instead of duplicating it per panel. Reuses
`load_rows`/`aggregate`/`SERIES_COLORS` from `plot_comprehension_results.py`
rather than re-deriving the aggregation or palette logic. Output at
`results/summary_capacity_and_duration.png`.

## Next steps

- Replicate either sweep (object count or tracking duration) on a second
  model to see whether these effects are qwen-specific or general --
  still an open item from prior sessions.
- `n_cued` and `force_path_crossing` remain unswept axes.
- The "increasing cued circles" half of today's original TODO title was
  explicitly skipped per the user, in favor of the summary figure -- still
  on the table as a future sweep if wanted.
