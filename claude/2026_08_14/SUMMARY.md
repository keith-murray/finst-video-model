# Summary: building and running the two Pylyshyn-motivated tasks

Full context/goals: `claude/2026_08_14/TODO.md`, following on from
`claude/2026_08_12/TODO.md`'s two-questions reorg.

## 1. Built the pylyshyn arm

New `src/finst_video_model/comprehension/pylyshyn/` (`config.py`,
`physics.py`, `stimulus_gen.py`), siloed from `comprehension/` but reusing
its shared `vlm_client.py`/`scoring.py`. Reproduces Pylyshyn's original
FINST/MOT paradigm more faithfully than the pre-existing smooth-tracking
task: 10 white crosses on black, 1-5 cued by blinking while stationary,
then all 10 move independently with direction/speed re-randomized on a
compass-heading grid and a continuous minimum-separation constraint
(elastic collision response) preventing ambiguous close encounters. Since
VLMs can't give a live keypress like Pylyshyn's human subjects, the report
is adapted to a single end-of-trial event: one object (target or
distractor, per `probe_on_target`) briefly turns into a solid square, and
the model answers True/False on whether it was cued at the start.
`scripts/pylyshyn/generate_samples.py` renders sample videos for visual QA.

**Adapted for ~1fps VLM sampling.** OpenRouter video-understanding backends
(e.g. Gemini) effectively downsample video to ~1 frame/sec regardless of
render fps. The original design (83ms probe flash, 1Hz cue blink, 0.2-0.5s
motion redirects) would have been almost entirely invisible at that
sampling rate, or worse, aliased. Reworked: `probe_flash_s` to 2.0s
(long enough to be caught by a sample regardless of phase), cue blink to
4x-oversampled 0.125s (decorrelates from the sampling phase instead of
aliasing with it, `cue_s` extended to 8.0s so this reliably shows up),
and the redirect cadence fixed at exactly 1.0s (`redirect_min_s ==
redirect_max_s`) so each VLM-perceived "step" is one explicit, sized
motion draw rather than an unpredictable sum of sub-second legs. Verified
by decoding real frames back out (blink toggling, probe hold duration,
per-second hop displacement) rather than trusting the math alone -- caught
nothing wrong here, but this same frame-by-frame verification habit is
what caught the bug below.

## 2. Reformatted the old smooth-tracking task

Moved `config.py`/`physics.py`/`stimulus_gen.py` from `comprehension/` into
`comprehension/smooth_pursuit/`, and changed its behavior to mirror
pylyshyn's report format while keeping its defining feature (one constant
heading per circle for the whole trial, not pylyshyn's frequent redirects):
cueing now happens before motion starts (circles stationary during the red
cue phase, previously they were already moving), a continuous
minimum-separation constraint now runs during tracking (previously only
enforced at initial placement), and the old end-of-clip letter-labeling
report was replaced with the same single True/False probe design as
pylyshyn (freeze, recolor one circle red, ask if it was cued at the
start). `scripts/smooth_pursuit/generate_samples.py` mirrors the pylyshyn
sample script.

**Bug caught during verification:** the first draft of `_draw_frame` left
cued circles colored red all the way through the tracking phase, not just
during cue/probe, because of a stray fallback condition. Found by
programmatically scanning decoded frames for red pixels across the
tracking-phase frame range; fixed with an explicit `show_cue` flag.

Known, accepted breakage: `scripts/prototype/run_comprehension_trial.py`,
`run_comprehension_batch.py`, and `analyze_position_heuristic.py` import
the old `comprehension.config/physics/stimulus_gen` paths directly and now
fail -- left as-is, same "historical record" status as the already-abandoned
generation arm's prototype scripts.

## 3. d' scoring

Added to the shared `comprehension/scoring.py` (not siloed -- both arms
produce the identical `probe_is_target` ground-truth shape): `
parse_boolean_answer` (extracts True/False from free text, `None` if
ambiguous), `classify_trial` (hit/miss/false_alarm/correct_rejection),
and `compute_d_prime` (pools many trials into hit-rate/false-alarm-rate,
applies the Hautus 1995 log-linear correction so a 0%/100% rate in a small
sample doesn't blow up to +-infinity, returns `d_prime` and the companion
bias metric `criterion`). Added `scipy` as a new dependency for
`scipy.stats.norm.ppf` (the inverse normal CDF the formula needs).
Verified with synthetic hand-checked cases before ever touching real data.

## 4. Single-trial scripts, live-smoke-tested

`scripts/pylyshyn/run_pylyshyn_trial.py` and
`scripts/smooth_pursuit/run_smooth_pursuit_trial.py`: render one trial's
video, build the True/False question text, call the VLM via
`vlm_client.ask_about_video`, score with `parse_boolean_answer`/
`classify_trial`. Both were run live against `google/gemini-2.5-flash`
during development (a few cents total) and caught/fixed singular/plural
question-grammar bugs at `n_cued=1` in the process.

## 5. Ran the actual experiments

Per the TODO's "Let's run some simple experiments": both arms at their
simplest cueing condition (pylyshyn 1/10 cued, smooth pursuit 1/5 cued),
20 seeds each, model `google/gemini-3.7-flash` pinned to the Google Vertex
provider specifically (confirmed via OpenRouter's `/models/{id}/endpoints`
to be exactly half the price of the default AI Studio provider) with
reasoning effort capped at `"minimal"` (the cheapest allowed setting --
this endpoint 400s if reasoning is disabled outright, and even
`"minimal"` isn't a hard token cap; Google decides the actual reasoning
spend internally regardless of the requested effort).

Added optional `provider`/`reasoning` passthrough params and a
`return_usage` flag to `vlm_client.ask_about_video` to support this
(backward compatible, defaults unchanged). Each of the 20 seeds was run
twice -- once with the probe on a cued target, once on a distractor --
sharing an identical underlying scene each time (placement/cueing/motion
all derive from `seed` alone; only which object gets probed differs),
giving a paired 40-trial design per arm (80 trials total) with both
conditions d' needs. `scripts/pylyshyn/run_pylyshyn_batch.py` and
`scripts/smooth_pursuit/run_smooth_pursuit_batch.py` run this, writing
`results/<batch_id>/{config.json,results.csv}` (git-tracked) and
`data/<batch_id>/trials/<trial_id>/` (gitignored raw artifacts) --
incremental writes + `--resume` support the archived batch-runner
convention.

Cost was estimated before running (~$0.001/trial baseline turned out not
to transfer to this model -- Gemini 3.7 Flash's default reasoning made a
single unconstrained probe trial cost $0.0089, dropped to ~$0.002-0.003
with `reasoning: {"effort": "minimal"}`), giving a ~$0.20 estimate for the
full 80-trial batch. Actual billed cost came in slightly under that: **$0.18**
total ($0.113 pylyshyn batch `fd799482`, $0.067 smooth-pursuit batch
`e97a6d18`).

`scripts/plot_confusion_and_dprime.py` (shared across both arms) renders
each batch's `results.csv` into a two-panel figure -- a 2x2 confusion
matrix (sequential-blue heatmap) and a hit-rate/false-alarm-rate bar panel
annotated with d'/criterion (categorical blue/orange) -- following the
project's dataviz color-by-job convention. Smoke-tested against synthetic
data before spending anything on the real batches.

### Results

| | Pylyshyn (1/10 cued) | Smooth pursuit (1/5 cued) |
|---|---|---|
| Hit rate | 0.98 (20/20 target trials) | 0.50 (10/20) |
| False-alarm rate | 0.50 (10/20 distractor trials) | 0.17 (3/20) |
| d' | 1.98 | 0.97 |
| criterion c | -0.99 (biased toward "True") | +0.48 (biased toward "False") |

Both models show real above-chance sensitivity (d' > 0 in both), and
pylyshyn's is roughly double smooth pursuit's. The more interesting
finding is the opposite response bias between the two tasks: on pylyshyn
the model says "True" almost regardless of the actual probe (perfect hit
rate, but a coin-flip false-alarm rate too), while on smooth pursuit it
leans the opposite way, toward "False." Raw accuracy alone would have
told a misleading and inconsistent story in each direction -- exactly the
failure mode d'/criterion decomposition exists to catch.

Figures: `results/fd799482/confusion_and_dprime.png` (pylyshyn),
`results/e97a6d18/confusion_and_dprime.png` (smooth pursuit).

## Not yet done

- Neither `results/fd799482` nor `results/e97a6d18` is indexed in a
  `results/INDEX.md` yet (the existing one, `results/prototype/INDEX.md`,
  only covers the archived prototype batches).
- No sweep beyond the single simplest-condition point per arm -- e.g.
  varying `n_cued`/field size, comparing models/providers, or a larger
  seed count for tighter d' confidence intervals.
- No statistical uncertainty (e.g. bootstrap CI) reported on d' itself,
  only the point estimate.
