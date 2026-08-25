# Summary: testing Qwen 3.8 on the pylyshyn task

Full context/goals: `claude/2026_08_19/TODO.md` -- coming back to the
project after a 3-week gap (`claude/2026_08_12/TODO.md`), pivoting from
Gemini 3.7 Flash (closed-source, used in `claude/2026_08_14/SUMMARY.md`'s
results) to Qwen 3.8's two variants, `qwen/qwen3.8-27b` (open-weight) and
`qwen/qwen3.8-max` (flagship, weights not released), since mechanistic
interpretability needs open weights.

## 1. Cost estimation

Live OpenRouter pricing (`/api/v1/models`, `/models/{id}/endpoints`) plus a
single live pilot trial per model (per [[feedback-verify-pricing]] -- don't
trust doc examples) gave real per-trial costs: `qwen3.8-27b` ~$0.006/trial
at reasoning effort `"low"`, `qwen3.8-max` ~$0.031/trial even at its
`"minimal"` reasoning floor (mandatory reasoning, unlike Gemini there's no
cheaper alternate provider worth pinning -- checked, <15% spread). Grid
size confirmed at 9 conditions (n_objects in {2,4,6,8,10} x n_cued
1,3,5,... up to half of n_objects) x 20 trials/condition (10
matching + 10 non-matching, distinct seeds, not each seed run twice like
the Aug-14 batches) x 2 models = 360 trials, ~$6.68 estimated total.

## 2. Batch runner: n_objects x n_cued x model grid

`scripts/pylyshyn/run_pylyshyn_batch.py` rewritten from its Aug-14
single-condition form (`n_cued` fixed at 1, `n_objects` fixed at 10) into a
full grid runner: `default_n_cued_values(n_objects)` derives the 1,3,5,...
sequence per the TODO's rule exactly, `build_grid` crosses it with
`n_objects` and a seed list split into matching/non-matching halves, and
`MODEL_CONFIGS` maps each model slug to its pinned `reasoning` setting so
multiple models can share one run (`model` becomes a `results.csv` column).
`--models`/`--n-objects`/`--seeds`/`--concurrency` are all CLI-overridable.

**Directory scheme changed from a random `batch_id` to a deterministic,
human-readable one**: `results/pylyshyn/<run_name>/` and
`data/pylyshyn/<run_name>/`, where `run_name` defaults to the model
slug(s) joined by `_` (e.g. `qwen3.8-27b`) but is overridable with
`--run-name` -- needed whenever the same model is re-run under a
different, non-`MODEL_CONFIGS`-default setting (see the reasoning-effort
comparison below), since the dedup key is `(model, n_objects, n_cued,
seed, probe_on_target)` and does *not* include the reasoning config, so
two different-config runs of the same model would otherwise silently
collide in one `results.csv`.

## 3. Two bugs found and fixed while running the first full batch

**Bug 1 -- unhandled `None` response content crashed the whole batch.**
The first full `qwen3.8-27b` run (109/180 trials in, at reasoning
`"low"`) crashed with `AttributeError: 'NoneType' object has no attribute
'lower'`: the model occasionally spent its entire completion-token budget
on reasoning and returned `content: null`, and `vlm_client.ask_about_video`
let that `None` propagate instead of raising something the batch runner's
`except` clause could catch. Fixed: `ask_about_video` now raises a
`RuntimeError` (with `finish_reason` and `usage` for debugging) when
content is `None`, so a single bad response becomes one `error` row
instead of killing the process. This first run was discarded per the
user (~$1 spent, called acceptable) once the reasoning-setting question
below made it moot anyway.

**Bug 2 -- seed relabeling from mixing different `--seeds` subsets.**
`build_grid`'s matching/non-matching split derives from a seed's *position*
within the whole `--seeds` list (sorted, first half True), so a quick
2-seed smoke test (`--seeds 0 1`) and the real 20-seed run assign
different `probe_on_target` labels to the same seed value (seed=1 is
`False` in a 2-seed list but `True` in the 0-19 list) -- both got written
to the same `results.csv` since they're different dedup keys, producing a
stray 181st row for one condition. Hit this exact bug twice (once each for
`qwen3.8-27b` and `qwen3.8-27b-low`) before adding a guard: `main()` now
errors out if `--seeds` doesn't exactly match the seed list already stored
in that run directory's `config.json`, rather than silently accepting a
mismatched subset. Both stray rows were manually stripped from
`results.csv` (and their `data/.../trials/<trial_id>/` artifacts removed)
before the guard existed to catch future instances.

Both full batches also hit a handful of transient `finish_reason='error'`
responses (cost $0 each) -- backfilled per the established convention:
strip the `error` rows from `results.csv`, re-run the same command (dedup
skips everything else).

## 4. Reasoning-effort matters more than model choice

`qwen3.8-27b`'s reasoning is optional (`qwen3.8-max`'s is mandatory --
confirmed live, disabling it 400s with `"Reasoning is mandatory for this
endpoint and cannot be disabled"`), so the first full `27b` run used
`reasoning: {"enabled": False}` for cost/reliability. Result: pooled
d' = **-0.22** (at/below chance), noisy per-condition (-0.7 to +0.9, no
pattern) -- `results/pylyshyn/qwen3.8-27b/`, 180 trials, **$0.93**.

An 18-trial spot check (same trials, re-run at `reasoning: {"effort":
"low"}`) found this was a reasoning-effort artifact, not a real capability
ceiling: d' jumped from **-1.71** (disabled, on this subset) to **+0.77**
("low"), with 11/18 answers flipping. Full re-run at `"low"` confirmed it:
`results/pylyshyn/qwen3.8-27b-low/`, 180 trials, **$1.87**, pooled
d' = **+1.30**, every one of the 9 conditions positive (0.23 to 1.84),
accuracy 55-85% throughout.

`qwen3.8-max` at its mandatory `"minimal"` reasoning floor:
`results/pylyshyn/qwen3.8-max/`, 180 trials, **$5.65** (~6x
`27b`'s per-trial cost), pooled d' = **0.07** -- also at chance, noisy
per-condition (-0.62 to +0.59), no pattern.

**Headline finding**: the smaller, cheaper `qwen3.8-27b` with reasoning on
clearly outperforms both its own no-reasoning version and the larger
"flagship" `qwen3.8-max` (stuck at its mandatory-minimal floor) -- scale
did not predict performance here, reasoning effort did. Whether
`qwen3.8-max` would also show a real signal at a higher reasoning effort
is an open question this session didn't answer (no cheaper way to test it
above `"minimal"`, and higher efforts would cost several dollars more per
condition at its pricing).

## 5. Summary figure

`scripts/pylyshyn/plot_qwen_summary.py` -- per the TODO's exact spec:
percent error (100 - accuracy, unparseable counted wrong) on the y-axis,
`n_cued` on the x-axis, 5 panels (one per `n_objects`), a dashed 50%
chance line. Three lines, not two: `qwen3.8-27b` appears at both reasoning
settings since that comparison is itself part of the story, plus
`qwen3.8-max`. Reuses the project's validated categorical palette slots
1-3 (blue/orange/aqua) and `scripts/prototype/plot_summary_figure.py`'s
shared-legend-across-panels convention. Confirms the headline finding
visually: `qwen3.8-27b` (low reasoning, blue) beats chance in every panel
and beats both other lines everywhere; `qwen3.8-max` (orange) and
`qwen3.8-27b` (no reasoning, green) are both flat-to-bad with no
n_objects/n_cued trend. Output:
`results/pylyshyn/qwen38_summary_percent_error.png`.

## Cost today

Precisely tracked: $0.037 (pilot cost-estimation calls) + $0.93
(`qwen3.8-27b`, no reasoning) + $5.65 (`qwen3.8-max`) + $0.17 (18-trial
reasoning spot check) + $1.87 (`qwen3.8-27b-low`) = **~$8.67**, plus the
discarded first `qwen3.8-27b` run (109 trials before the crash, not
precisely logged -- roughly $1 more by the user's own estimate at the
time). **Total for the day: ~$9-10.**

## Not yet done

- No re-run of `qwen3.8-max` at a higher (paid, non-mandatory-floor)
  reasoning effort, to check whether its near-chance result is also a
  reasoning-effort artifact rather than a real ceiling.
- The smooth-pursuit arm wasn't touched today -- only pylyshyn, per the
  TODO's explicit scope ("We will focus on the Pylyshyn task today").
- No statistical uncertainty (e.g. bootstrap CI) on any of today's d'/
  percent-error figures, same gap noted in `claude/2026_08_14/SUMMARY.md`.
- `results/pylyshyn/qwen3.8-27b/` (no-reasoning) is now mostly of interest
  as the reasoning-effort comparison point, not as a standalone result --
  worth a docstring/README note if it's confusing later which of the three
  `qwen3.8-27b*` directories is the "real" one (`qwen3.8-27b-low`).
