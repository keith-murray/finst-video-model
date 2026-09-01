# Summary: qwen3.6-plus replicates gemma's V-shaped debug_circular curve

Full context/goals: `claude/2026_09/2026_09_01/TODO.md`.

## Task 1: qwen3.6-plus on debug_circular's angle sweep

Motivation: `qwen3.6-plus` (a strong, much larger model than
`gemma-4-31b-it`, and non-reasoning) tied gemma on pylyshyn but was never
tested on debug_circular. The concern going in: is gemma's V-shaped,
below-chance-in-the-middle accuracy curve from
`claude/2026_08/2026_08_27/SUMMARY.md` a gemma-specific weakness (maybe a
"nearest neighbor" position heuristic, given pylyshyn should be strictly
harder than debug_circular's constant-velocity rotation), or something
about the stimulus itself?

Generalized `scripts/debug_circular/run_debug_circular_angle_sweep.py` and
`plot_angle_sweep.py` to take `--model` as a CLI arg (was hardcoded to
`google/gemma-4-31b-it`), mirroring the `--n-objects` generalization from
last session. `plot_angle_sweep.py` now reads the model name for its title
from `config.json` instead of a hardcoded string.

Live-priced `qwen/qwen3.6-plus` before running (per
[[feedback-verify-pricing]]): $0.325/M prompt, $1.95/M completion. A
4-trial smoke test under a throwaway run-name (deleted after, per
[[project-status-2026-08-27]]'s seed-lock gotcha -- never smoke-test under
the real run-name) gave ~$0.53 estimated for the full 360-trial run, and
surfaced that **qwen3.6-plus's prompt tokens *do* respond to the
duration-stretch trick** (1,652 -> 7,412 tokens native->stretched, ~4.5x),
unlike gemma's confirmed fixed-frame-budget insensitivity
([[reference-gemma-fixed-frame-budget]]).

Full run: `results/debug_circular/angle_sweep_qwen3.6-plus/` (`n_objects=3`,
`rotation_deg` in {40,80,...,360}, native+stretched, 20 seeds/condition,
360 trials, 0 errors, 0 unparseable).

| rotation_deg | native acc | stretched acc |
|---|---|---|
| 40 | 85% | 100% |
| 80 | 70% | 55% |
| 120 | 45% | 40% |
| 160 | 30% | 35% |
| 200 | 20% | 15% |
| 240 | 25% | 10% |
| 280 | 30% | 25% |
| 320 | 85% | 85% |
| 360 | 100% | 100% |

**Finding: the V-shape replicates almost exactly**, including going
*below chance* in the same 200-240 deg region gemma dipped in (10-25% vs.
gemma's 10-15%), and the same near-ceiling recovery at 40/320/360 deg.
Native and stretched track each other closely at every point despite the
prompt-token jump on stretched -- more real frame access didn't translate
into better performance for this model on this task either, matching
[[project-status-2026-08-27]]'s "no stretch benefit on interpolable
rotation" pattern.

This is a meaningful update on the open question from
[[project-status-2026-08-27]]: two very different models (91% on
pylyshyn's `qwen3.6-plus` vs. tied-91% `gemma-4-31b-it`, different
vendors/architectures/scales, one whose tokenization responds to the
stretch trick and one that doesn't) produce nearly the *same* accuracy
curve shape on debug_circular. That argues the V-shape (easy near 0/360
deg loop closure, hard near 180 deg antipodal rotation, below chance in
the trough) is a property of the **stimulus/task**, not a gemma-specific
heuristic or weakness -- weakens the original "gemma might be doing
position-matching, and that happens to work on pylyshyn" framing that
motivated testing it here, since qwen3.6-plus shows the identical pattern
without needing that explanation.

## Task 2: qwen3.8-27b (no reasoning) as a third data point

Same protocol, `qwen/qwen3.8-27b`, `reasoning={"enabled": False}` (matching
`run_pylyshyn_stretch_sweep.py`'s `"none"` level convention exactly --
`reasoning_param("none") == {"enabled": False}`). Smoke-tested first
(reasoning_tokens=0 confirmed reasoning genuinely off), ~$0.6-0.7 est. for
the full run. `results/debug_circular/angle_sweep_qwen3.8-27b/`, 360
trials, 0 errors.

| rotation_deg | native acc | stretched acc |
|---|---|---|
| 40 | 50% | 60% |
| 80 | 40% | 60% |
| 120 | 35% | 40% |
| 160 | 40% | 60% |
| 200 | 45% | 55% |
| 240 | 30% | 25% |
| 280 | 30% | 45% |
| 320 | 35% | 50% |
| 360 | 65% | 80% |

**No clean V-shape here** -- accuracy hovers near chance (25-65%)
across the entire angle range, with no near-ceiling recovery at the easy
40 deg endpoint (50/60% vs. gemma's 90/85% and qwen3.6-plus's 85/100% at
the same angle) the way both other models showed. This doesn't cleanly
confirm or disconfirm the V-shape-is-a-task-property reading from Task 1
-- `qwen3.8-27b` with reasoning off was already known to be a weak
performer on this task type ([[project-status-2026-08-20]]'s pylyshyn
"none" level, [[project-status-2026-08-26]]'s local nothink d'=0 "always
False" bias), so the more likely read is that this model simply isn't
capable enough overall to clear the easy cases and reveal the pattern --
not evidence against the V-shape itself. The V-shape may specifically
require a model strong enough to nail the 0/360 deg loop-closure cases
before the antipodal-region difficulty becomes visible as a *relative*
dip rather than just uniform noise.

## Task 3 (TODO's Task 2): how well would a "nearest neighbor" position heuristic do on pylyshyn?

Motivated directly by Task 1/2's V-shape findings: if debug_circular's
V-shape is a stimulus property (not a gemma-specific heuristic, per Task
1's finding), the concern flips around -- maybe the strong pylyshyn scores
from [[project-status-2026-08-27]] (qwen3.6-plus/gemma-4-31b-it near the
top of the 5-model comparison) reflect the *same* underlying heuristic
succeeding on pylyshyn, not genuine continuous identity tracking through
the random-walk motion.

New `scripts/pylyshyn/nearest_neighbor_heuristic.py`: since pylyshyn's
motion is fully deterministic given a trial's saved `TrialConfig` (all
randomness seeded off `cfg.seed`, never off the trial's uuid), re-simulates
each trial's object positions using the exact same `build_objects`/
`step_tracking_frame` physics the stimulus itself uses -- no video, no API
calls, no cost. The heuristic: remember only the single cued object's
*original* (pre-motion) position, then at probe time answer "match" iff
the flashing probed object is currently the spatially closest of the
`n_objects` to that remembered spot -- blind to actual object identity,
purely a proximity guess. Deduplicates native/stretched rows sharing a
seed (physically identical trial content, different trial_id) before
simulating, so each real trial is only counted once.

New `scripts/pylyshyn/plot_speed_sweep_models_heuristic.py`: re-draws just
the qwen3.6-plus (`results/pylyshyn/speed_sweep/`) and gemma-4-31b-it
(`results/pylyshyn/speed_sweep_models/`) rows of the existing 4-panel
(redirect x speed) grid, adding a red dashed horizontal line per panel for
the heuristic's accuracy on that model's own literal trial set. Written to
a new file, `results/pylyshyn/speed_sweep_models/accuracy_summary_heuristic.png`,
rather than overwriting the 5-model original.

| redirect | speed | qwen3.6-plus native/stretched | heuristic | gemma native/stretched | heuristic |
|---|---|---|---|---|---|
| slow (2.0s) | slow (16-33) | 94% / 94% | **94%** | 100% / 100% | **90%** |
| slow (2.0s) | fast (32-66) | 88% / 75% | **88%** | 75% / 80% | **80%** |
| fast (1.0s) | slow (16-33) | 100% / 100% | **94%** | 95% / 95% | **90%** |
| fast (1.0s) | fast (32-66) | 88% / 94% | **94%** | 90% / 95% | **85%** |

**The heuristic alone tracks almost exactly with both models' actual
accuracy in every single condition** -- pooled across the 4 conditions,
qwen3.6-plus averages ~92.5% vs. the heuristic's ~92.5%, and gemma-4-31b-it
averages ~91% vs. the heuristic's ~86%. This directly confirms the user's
concern: at this stimulus's current settings (`n_objects=3`, these
redirect/speed ranges), **a purely spatial, identity-blind proximity guess
performs about as well as either model**, meaning these pylyshyn scores
can't be read as strong evidence of genuine continuous object-identity
tracking through the motion -- the task doesn't currently separate that
capability from the cheaper heuristic. This significantly qualifies
[[project-status-2026-08-27]]'s "gemma-4-31b-it ties qwen3.6-plus for the
best pylyshyn score" framing: both may be closer to matching a
non-tracking heuristic's ceiling than to demonstrating FINST-style
identity tracking.

## Task 4 (TODO's Task 3): scaling n_objects to disambiguate tracking from the heuristic -- a null result

Direct follow-up to Task 3's finding: if the heuristic explains the strong
pylyshyn scores, adding more distractors should hurt the heuristic (more
objects competing to randomly land near the target's remembered original
position) more than it hurts genuine identity tracking. Restricted to the
slow-redirect/slow-speed condition per the user's explicit request (both
models' strongest condition, so any gap opening up is attributable to
`n_objects` alone).

Generalized `run_pylyshyn_speed_sweep_models.py` in place with `--run-name`/
`--n-objects`/`--redirect-conditions`/`--speed-conditions` (mirroring the
debug_circular angle_sweep_n4 precedent, including fixing the resume key to
include `n_objects`), and added `qwen/qwen3.6-plus` to its `MODEL_REASONING`
dict (previously only in the separate `run_pylyshyn_speed_sweep.py`).
Smoke-tested at `n_objects=5` first, including eyeballing a rendered frame
to confirm 5 well-separated, non-overlapping crosses render correctly at
384x384. Two runs, `nobjects_sweep_n4`/`nobjects_sweep_n5` (separate
run-names, not one shared one, to avoid the stale-`config.json` gotcha a
shared run-name would hit -- see this file's own module docstring update),
qwen3.6-plus + gemma-4-31b-it, 20 seeds/condition, 160 trials total, 0
errors.

New `scripts/pylyshyn/plot_nobjects_sweep.py` ->
`results/pylyshyn/nobjects_sweep/accuracy_by_nobjects.png`: 2 panels (one
per model), x-axis `n_objects` in {3,4,5} (n=3 pulled from the existing
`speed_sweep`/`speed_sweep_models` data), native/stretched lines, heuristic
overlay.

| n_objects | qwen3.6-plus native/stretched | heuristic | gemma native/stretched | heuristic |
|---|---|---|---|---|
| 3 | 94% / 94% | 94% | 100% / 100% | 90% |
| 4 | 90% / 90% | 95% | 90% / 85% | 95% |
| 5 | 85% / 90% | 90% | 85% / 85% | 90% |

**Null result -- the heuristic does not degrade as n_objects scales up.**
Both models decline mildly and roughly in parallel with the heuristic
(qwen3.6-plus 94%->~88%, gemma 100%->~85%), and the heuristic itself stays
flat at 90-95% throughout, even slightly *exceeding* model accuracy at
n_objects=4-5 in a couple of cells. Scaling `n_objects` alone, at this slow
condition, did **not** create the separation the hypothesis predicted.

Working explanation (not yet tested): at `redirect_s=2.0`/`speed_px_s=
16-33`, absolute displacement by the time of the probe is small relative to
the 384x384 field regardless of how many objects share the field -- so how
often *some* distractor randomly wanders near the cued object's original
spot is governed mainly by motion speed/duration, not by how many
distractors exist. This would predict the heuristic-vs-tracking gap should
open up under the **fast** speed/redirect conditions (where
[[project-status-2026-09-01]]'s Task 3 numbers already show larger raw
displacement) rather than under slow motion -- worth re-running this same
`n_objects` scaling at `--redirect-conditions fast --speed-conditions fast`
as the next attempt, rather than concluding real tracking and the heuristic
are simply inseparable on this stimulus.

## Task 5: retrying the n_objects scaling at fast redirect/speed

Per the user's direct follow-up ("try the fast redirect/speed condition
next"): re-ran the same `n_objects` in {3,4,5} scaling, this time at
`redirect_s=1.0`/`speed_px_s=32-66` (both fast), testing the working
explanation from Task 4's null result -- that a heuristic-vs-tracking gap
should be more visible where objects displace further by probe time.
Reused the same `nobjects_sweep_n4`/`nobjects_sweep_n5` run-names (a second
invocation each, `--redirect-conditions fast --speed-conditions fast`) --
safe since the resume key already includes redirect/speed condition, though
note each run-name's `config.json` metadata still only reflects whichever
invocation wrote it first (a known, accepted staleness, same class as prior
sessions' model-field staleness). 160 more trials, 0 errors.

Generalized `plot_nobjects_sweep.py` with `--condition {slow,fast}` (was
hardcoded to slow); regenerated the slow-condition figure under a clearer
name (`accuracy_by_nobjects_slow.png`, replacing the old
`accuracy_by_nobjects.png`) and added `accuracy_by_nobjects_fast.png`.

| n_objects | qwen3.6-plus native/stretched | heuristic | gemma native/stretched | heuristic |
|---|---|---|---|---|
| 3 | 88% / 94% | 94% | 90% / 95% | 85% |
| 4 | 70% / 85% | 75% | 80% / 85% | 75% |
| 5 | 75% / 75% | 75% | 90% / 90% | 75% |

**Mixed, partially-supportive result -- more encouraging than Task 4's
slow-condition null, but noisy at n=20/condition.** For `gemma-4-31b-it`,
the heuristic now visibly degrades (85%->75%->75%) while gemma's own
accuracy holds flat or even recovers (90%->80%->90%), opening a growing gap
(5pp -> 5pp -> 15pp) that didn't exist at slow speed. For `qwen3.6-plus`
the picture is messier: native accuracy at n_objects=4 (70%) actually falls
*below* the heuristic (75%), while stretched stays above it at n=3/4 (94%/
85% vs. 94%/75%) but the gap fully closes at n_objects=5 (both exactly
75%). At n=20/cell the SEM on a single proportion is roughly +-10pp, so
several of these differences (especially qwen's) are within noise -- this
reads as a directionally promising but not yet confirmed effect, strongest
for gemma, not a clean resolution of Task 3/4's open question. More
seeds/condition would be needed to say anything with confidence.

## Token-math estimate of gemma's frame budget (before committing to local hosting)

User flagged gemma's 90% at pylyshyn n_objects=5 (fast/fast) -- 15 points
above qwen3.6-plus on identical trials -- as "mysterious," wanting to know
how much gemma is undersampling, floating local-hosting as next session's
plan. Tried the cheap thing first: `gemma-4-31b-it`'s `prompt_tokens` is
constant (2528-2530 across 80 pylyshyn native trials checked; 2541 on
debug_circular, per [[reference-gemma-fixed-frame-budget]]). Estimated the
text-question token count with `tiktoken`'s `cl100k_base` (a stand-in for
Gemma's real tokenizer -- not exact, but a reasonable approximation) and
subtracted it out:

| stimulus | prompt_tokens | text tokens (cl100k_base) | video residual |
|---|---|---|---|
| pylyshyn (n_objects=3) | 2528 | 179 | 2349 |
| debug_circular (n_objects=3) | 2541 | 192 | 2349 |

**The residual matches exactly across two completely different question
texts** -- strong internal consistency that this really isolates the pure
video/image token budget. If `gemma-4-31b-it` shares Gemini's documented
~258-tokens/frame image tokenization convention (same vendor lineage,
*not* confirmed for this specific model), 2349/258 ≈ **9.1 frames** --
i.e., gemma is likely sampling only ~9 frames total from these ~100-frame/
10s clips, regardless of real duration (consistent with the existing
fixed-budget finding). Two explicit caveats: the per-frame-token-cost
figure is borrowed from a sibling product line, and the text-token count
uses a generic tokenizer, not Gemma's actual one -- so "~9 frames" is an
informed estimate to be confirmed or refuted by local hosting, not a
measurement.

If it holds up, this reframes gemma's n_objects=5 result in an interesting
way: it may not reflect more genuine continuous tracking than
qwen3.6-plus, but correspondence-matching over a much sparser snapshot
sequence -- which connects directly to the user's earlier "would a
10-frame stimulus reduce heuristic reliance" thought (gemma may already
effectively be running something like that experiment on its own, without
us designing it that way).

## Open threads (carried over / updated)

- The debug_circular V-shape is now replicated across gemma-4-31b-it
  (n_objects=3 and 4) and qwen3.6-plus (n_objects=3) -- three independent
  replications, two different models. `qwen3.8-27b` (no reasoning) shows
  no clean V-shape, but reads as "too weak overall to clear the easy
  cases" rather than a genuine counterexample -- worth testing a model of
  intermediate strength, or `qwen3.8-27b` with reasoning enabled (known
  much stronger per [[project-status-2026-08-27]]'s "high" finding), to
  see whether the V-shape reappears once a model clears ~80%+ at the easy
  endpoints. Its mechanism is still unexplained;
  [[project-status-2026-08-27]]'s suggestion of a direct rotation-magnitude
  estimation task (vs. this binary same/different judgment) or a
  finer-grained angle sweep remains the natural next step.
- `smooth_pursuit` arm still untouched since 2026-08-14.
- `qwen3.6-plus` joins `qwen3.8-27b`/`qwen3.5-397b-a17b` as a model whose
  tokens respond to the duration-stretch trick, but (like those models on
  pylyshyn, and unlike qwen3.8-27b/397b-a17b on debug_circular) the extra
  tokens didn't produce an accuracy benefit here -- worth remembering this
  isn't a reliable free win even when a model's tokenizer does respond to
  it.
- **The pylyshyn nearest-neighbor heuristic result is still the most
  important open thread.** Slow redirect/speed showed no gap (Task 4,
  null); fast redirect/speed (Task 5) showed a directionally promising but
  noisy gap, clearest for gemma-4-31b-it (heuristic degrading while gemma
  holds flat), messier for qwen3.6-plus. At n=20/condition this isn't
  confident yet -- the natural next step is more seeds/condition at fast
  redirect/speed specifically (both models), before drawing a firm
  conclusion either way. If a real, well-powered gap confirms there, that
  would be the first evidence this stimulus *can* separate real tracking
  from the heuristic under the right motion conditions. A
  heuristic-adversarial probe placement (deliberately place a probe near
  the target's original position when it's NOT the target) remains a
  fallback lever if more seeds don't clarify things. Also cheap to extend
  this same heuristic check to the other 3 models in
  `speed_sweep_models/accuracy_summary.png` (kimi-k3, qwen3.5-397b-a17b,
  glm-5v-turbo) -- the heuristic-accuracy ceiling is model-independent and
  already implemented, no new API spend needed.
- The user separately floated (not yet implemented, explicitly "just to
  think about"): is 100 frames too long a context for genuine tracking,
  such that a much shorter clip (~10 frames, larger objects so trajectories
  can still be "complex" within that span) would show *less* heuristic
  reliance? Flagged risk if this is ever tried: sparse frames turn
  continuous tracking into snapshot-to-snapshot correspondence matching,
  which a frame-to-frame nearest-neighbor matcher could solve just as
  easily as the current heuristic does, unless per-step displacement stays
  below the minimum object-object separation at every step -- that caps
  how complex the trajectory can be without reintroducing the same kind of
  shortcut. Also interacts with [[reference-gemma-fixed-frame-budget]]:
  gemma already extracts a fixed-size frame sample regardless of real
  video length, so a 10-frame clip might just give gemma full temporal
  resolution rather than "less to reason over," while for qwen3.6-plus
  (whose tokens scale with frame count) it would be a genuine context-size
  reduction -- the two models might respond quite differently to this
  manipulation for reasons unrelated to the tracking-vs-heuristic question
  itself.
