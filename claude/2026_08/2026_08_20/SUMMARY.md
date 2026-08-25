# Summary: reasoning-budget sweep on the simplest pylyshyn condition

Full context/goals: `claude/2026_08_20/TODO.md`, following on from
`claude/2026_08_19/SUMMARY.md`'s finding that reasoning effort mattered more
than model choice for `qwen3.8-27b` (only tested at "low" there).

## 1. Chose `reasoning.effort`, not `reasoning.max_tokens`, as the swept axis

Tried the literal-sounding option first: OpenRouter's `reasoning.max_tokens`
as a numeric token budget. A same-question pilot (text-only "what is 2+2")
and a same-video pilot (the real n_objects=2 stimulus) both showed it isn't
enforced as a hard cap -- `qwen3.8-27b` burned 654 reasoning tokens against a
requested budget of 100, and reasoning-token counts didn't track the
requested number at all (100 -> 654, 500 -> 915, 2000 -> 541). Switched to
`reasoning.effort` (minimal/low/medium/high, plus "none" for `qwen3.8-27b`
only, since `qwen3.8-max`'s reasoning is mandatory) -- a pilot on the real
video confirmed this one *is* respected in aggregate (`qwen3.8-max`'s
reasoning_tokens jumped from ~1000 at minimal/low to ~5000-6000 at
medium/high), even though per-trial variance is still real (`qwen3.8-27b`'s
`none` pilot for the effort-vs-tokens check, from the 08-19 batch, doesn't
strictly increase step to step either).

## 2. Batch runner and plot

New `scripts/pylyshyn/run_pylyshyn_reasoning_batch.py`, fixed at
n_objects=2/n_cued=1 (the TODO's "simplest version"), sweeping
`reasoning.effort` per REASONING_LEVELS_BY_MODEL (`qwen3.8-27b`: none,
minimal, low, medium, high; `qwen3.8-max`: minimal, low, medium, high --
mandatory reasoning confirmed live, "none" 400s). 20 seeds/condition (10
matching + 10 non-matching), same convention as `run_pylyshyn_batch.py`,
including its seeds-mismatch guard and incremental-write/`--resume` support.
results.csv also logs `reasoning_tokens` per trial (from
`usage.completion_tokens_details`) as a diagnostic, even though the plotted
x-axis is the ordinal effort label, not a token count.

3 trials hit the familiar transient `finish_reason='error'`
(reasoning-token budget exhausted, content=None) -- backfilled per the
established convention (strip error rows, re-run, dedup skips the rest).

`scripts/pylyshyn/plot_reasoning_summary.py` -- single-panel line plot,
x=reasoning effort level (ordinal), y=percent error, one line per model,
following `plot_qwen_summary.py`'s color/chrome conventions. Output:
`results/pylyshyn/reasoning_sweep/reasoning_sweep_percent_error.png`.

## 3. Cost

Estimated $3.85 from two live pilot calls before running (`qwen3.8-27b`
~$0.004-0.013/trial, `qwen3.8-max` ~$0.028-0.052/trial). Actual: **$3.91**
for all 180 trials.

## 4. Results

| | none | minimal | low | medium | high |
|---|---|---|---|---|---|
| `qwen3.8-27b` % error | 45 | 35 | 20 | **15** | 25 |
| `qwen3.8-27b` d' | +0.23 | +0.75 | +1.57 | **+2.16** | +1.22 |
| `qwen3.8-max` % error | -- | 40 | 45 | 45 | 45 |
| `qwen3.8-max` d' | -- | +0.46 | +0.23 | +0.24 | +0.23 |

**Headline finding, answering 08-19's open question**: `qwen3.8-max`'s
near-chance performance is *not* a reasoning-effort artifact -- it stays flat
at d' ~0.2-0.5 (45% error) across every effort level from minimal to high,
despite reasoning-token spend scaling ~5-6x over that range (see pilot
numbers above). This looks like a real capability ceiling for `qwen3.8-max`
on this task, not an under-resourced-reasoning ceiling.

`qwen3.8-27b`, by contrast, shows a real (if non-monotonic-at-the-top)
reasoning dose-response: d' rises from +0.23 (no reasoning) to a peak of
+2.16 at "medium," then dips to +1.22 at "high" -- more reasoning helped up
to a point, then didn't help further (single seed count per condition, n=20,
so this dip may not be a stable effect, just noise around a plateau).

## Not yet done

- Only one condition (n_objects=2, n_cued=1) was swept -- no check of
  whether the reasoning-budget effect holds at harder conditions (more
  objects/more cued), which is where `qwen3.8-27b`'s Aug-19 "low"-reasoning
  results were closer to chance.
- No bootstrap CI on any d'/percent-error point (same gap noted in both
  prior summaries) -- with n=20/condition here, the `qwen3.8-27b`
  medium-vs-high dip in particular is likely within noise.
- `qwen3.8-max`'s flat-at-chance result across all four effort levels is
  new evidence toward "real ceiling," but wasn't cross-checked against a
  higher-difficulty or differently-phrased probe to rule out a
  task-specific quirk (e.g. the question wording) rather than a genuine
  MOT-capacity ceiling.
