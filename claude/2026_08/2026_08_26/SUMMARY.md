# Summary: reasoning-effort sweeps, the OpenRouter duration-stretch trick, and pylyshyn revisited

Full context/goals: `claude/2026_08/2026_08_26/TODO.md` -- four parts, all
touched today.

## 1. Local cluster reasoning-effort sweep (Part 1)

Extended the debug_circular cluster pipeline
(`scripts/debug_circular/run_cluster_batch.py`,
`profile_cluster_timing.py`, `aggregate_cluster_results.py`,
`slurm/{run_debug_circular_batch,profile_debug_circular}.sh`) with a
`--reasoning-effort {low,medium,xhigh}` + `--preserve-thinking` flag pair,
reusing the existing 200-trial `qwen3.8-27b-nothink` video set for all
three effort levels rather than regenerating stimuli (response files are
now per-level, `cluster_response_<effort>.json`, coexisting with the
already-committed nothink baseline's `cluster_response.json`). Profiling
jobs were submitted but cluster queue wait times turned out to be
"horrendous," so this became a side thread for the rest of the session
-- **still not run to completion as of this summary**, see "Not yet done."

## 2. The OpenRouter duration-stretch trick (Part 2/3, debug_circular)

While waiting on the cluster, tested whether an OpenRouter-hosted model
could be tricked into keeping more video frames by re-encoding the same
100-frame debug_circular clip at a much lower mp4 fps (2 instead of 10),
stretching the nominal duration from 10s to 50s without changing any pixel
content. New `finst_video_model.debug_circular.stimulus_gen.write_mp4()`
(decoupled-fps encoder) + `scripts/debug_circular/generate_openrouter_samples.py`
(16 paired native/stretched samples) +
`scripts/debug_circular/run_openrouter_diagnostic.py` (sends both variants
to OpenRouter, compares `usage.prompt_tokens` and accuracy).

**It worked**: `usage.prompt_tokens` jumped ~4.3x (native ~1718 ->
stretched ~7441 on `qwen3.8-27b`), confirming OpenRouter's frame-sampling
is duration/fps-sensitive, not purely content-based. But the accuracy
payoff **depended entirely on reasoning effort**:

| reasoning | native d' | stretched d' |
|---|---|---|
| none | 0.97 | 0.00 (chance) |
| medium | 0.75 | **1.56** |

With reasoning off, more frames didn't help (accuracy collapsed to
chance); with `medium` reasoning on, stretched clearly beat native --
more than double, the best d' seen anywhere in this project's
debug_circular work, including the local no-thinking cluster baseline
(d'=0). `qwen3.8-max` hit 100% accuracy on both variants at both `low` and
`medium` reasoning (already at ceiling on native alone, so no room for the
trick to show a benefit there) -- **this result is scoped to debug_circular
only**: its rigid-ring rotation is solvable by interpolating a handful of
frames (closer to a math problem), unlike pylyshyn's random-walk motion,
so it doesn't imply anything about `qwen3.8-max` vs `qwen3.8-27b` more
broadly (see part 3 below, and `results/debug_circular/openrouter_diagnostic/`
for the full 5-condition x 2-model results.csv + `accuracy_summary.png`).

## 3. Pylyshyn ported to debug_circular's design + native-vs-stretched x reasoning sweep (Part 3)

Since the stretch trick worked, revisited `pylyshyn` (untouched since
2026-08-14). Ported its visual design to match `debug_circular` almost
exactly (`finst_video_model.pylyshyn.config`/`stimulus_gen.py`): 384x384,
same cross size, color-based cueing/probing (red, steady -- no more
blinking or a probe shape-change), `n_objects=3`/`n_cued=1` to start.
Motion itself (continuously-redirecting random walk) stayed pylyshyn's
own, per the user's correction that this is a genuinely different *kind*
of task from debug_circular's constant-velocity rotation, not just a
harder version of it.

Timing was then slowed at the user's explicit request to exactly match
debug_circular's 100-frame budget (`fps=10`, `cue_s=1.0`, `tracking_s=9.0`,
`probe_flash_s=1.0`), plus a "slow the motion down" pass (`redirect_s`
1.0->2.0s, `speed_px_s` roughly halved to 16-33) picked after generating
and eyeballing `scripts/pylyshyn/generate_samples.py` samples across four
speed scales. `generate_stimulus()` now returns `(ground_truth, frames)`
(was just `ground_truth`) so a caller can losslessly re-encode a stretched
variant from the same in-memory array, mirroring debug_circular's
`write_mp4()` reuse.

New `scripts/pylyshyn/run_pylyshyn_stretch_sweep.py`: model x
reasoning_effort x video variant (native/stretched), 16 seeds/condition
(8 matching + 8 non-matching). Along the way, chased what looked like a
`max_tokens`-truncation bug in `qwen3.8-27b`'s low/medium reasoning
(`usage.reasoning_tokens` pinned at a suspiciously constant ~65/~257
across dozens of trials) -- added `max_tokens` support to
`finst_video_model.vlm_client.ask_about_video` (kept, useful for its
ordinary purpose), but it didn't fix the pinning, and neither did the
alternative `reasoning.max_tokens` field (mutually exclusive with
`effort`, confirmed via the API's own 400 error; sweeping 100-4000 all
still landed at ~65, only 8000 broke through once). **Concluded this is a
real, reproducible property of `qwen3.8-27b`'s effort tiers on this
endpoint, not a bug** -- see
`reference-openrouter-reasoning-max-tokens` memory for the full
investigation.

Final results (`results/pylyshyn/stretch_sweep/accuracy_summary.png`,
`results.csv`):

| model | reasoning | native acc | stretched acc |
|---|---|---|---|
| qwen3.8-27b | none | 56% | 50% (chance) |
| qwen3.8-27b | low | 56% | 50% (chance) |
| qwen3.8-27b | medium | 62% | 62% |
| qwen3.8-max | low | 100% | 100% |
| qwen3.8-max | medium | 100% | 100% |

Unlike debug_circular, **the stretch trick shows no consistent benefit on
pylyshyn** -- native and stretched track each other closely at every
level, consistent with the "nothing to interpolate" distinction above.

## 4. Can a non-reasoning model solve it at all? (Part 4)

Tested whether reasoning is actually *necessary*, or whether `qwen3.8-max`
was just winning on model scale. Reused `run_pylyshyn_stretch_sweep.py`
unchanged -- added two more OpenRouter Qwen models (both confirmed
`reasoning.mandatory=false` via a live `/models` check) with
`reasoning={"enabled": False}`:

| model | native acc | stretched acc |
|---|---|---|
| qwen3.5-122b-a10b (122B MoE, only 10B active) | 75% | 75% |
| qwen3.6-plus (dense, tested as a follow-up once "A10B" was noticed) | **94%** | **94%** |

**Both non-reasoning models clearly beat every `qwen3.8-27b` reasoning
condition**, and `qwen3.6-plus` gets most of the way to `qwen3.8-max`'s
ceiling while spending nothing on chain-of-thought. Answers Part 4's
question: yes, a sufficiently capable model can solve this stimulus in one
feedforward pass -- `qwen3.8-27b`'s struggles look more like a
scale/capability gap than a reasoning-necessity one. A full pricing check
of all 16 video-capable Qwen models on OpenRouter (only `qwen3.8-max` has
mandatory reasoning) is in this session's chat if needed again -- not
saved verbatim to memory since OpenRouter pricing drifts.

## Not yet done

- **The local cluster reasoning-effort sweep (Part 1) never finished** --
  profiling jobs were submitted but queue waits were long enough that the
  session pivoted to the OpenRouter side quest for the rest of the day.
  Check job status and either resume or re-profile before running the real
  200-trial x 3-effort-level batch.
- `smooth_pursuit` arm still untouched since 2026-08-14.
- pylyshyn: `n_objects=3` now looks too easy for several models
  (`qwen3.8-max` at 100%, `qwen3.6-plus` at 94%) -- scaling it up is a
  natural next step. `effort="high"` untested for `qwen3.8-27b` (might
  unlock a genuinely larger reasoning budget beyond the pinned ~257-token
  medium tier). The Part 4 finding suggests a deliberate capability x
  reasoning grid (not just one point per model) would be worth running if
  this arm continues.
- The `debug_circular` vs `pylyshyn` divergence on the stretch trick
  (helps one, not the other) is explained qualitatively (interpolable vs.
  genuinely unpredictable motion) but not tested against a third stimulus
  type to confirm the pattern generalizes.
