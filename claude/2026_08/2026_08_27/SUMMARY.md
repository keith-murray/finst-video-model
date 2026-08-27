# Summary: filled out the stretch sweep, a new speed/redirect sweep, a 5-model pylyshyn comparison, and a debug_circular follow-up on the surprise model

Full context/goals: `claude/2026_08/2026_08_27/TODO.md` -- all four tasks
touched today, plus extra models added live mid-session on top of Task 3.

## 1. Filling out the stretch sweep with qwen3.8-27b at "high" reasoning (Task 1)

Added `"high"` to `REASONING_LEVELS_BY_MODEL` in
`scripts/pylyshyn/run_pylyshyn_stretch_sweep.py` (the TODO said "xhigh",
but that's the local-vLLM-only reasoning value -- OpenRouter's ladder here
is minimal/low/medium/**high**, confirmed with the user before running).
Unlike low/medium (pinned to a small ~65/~257-token reasoning budget),
`"high"` broke free and produced the best d' seen anywhere in this
project's pylyshyn work:

| level | native d' | stretched d' |
|---|---|---|
| none | 0.28 | 0.00 |
| low | 0.38 | 0.08 |
| medium | 0.59 | 0.56 |
| **high** | **2.56** | **3.19 (ceiling)** |

Also rewrote `scripts/pylyshyn/plot_stretch_sweep.py` from one subplot per
model (bar width scaled with panel count, so a 1-level model like
qwen3.6-plus got comically wide bars) to a single axes with one x-tick per
`(model, reasoning_level)` pair. `results/pylyshyn/stretch_sweep/accuracy_summary.png`
regenerated.

## 2. New speed/redirect sweep on qwen3.6-plus (Task 2)

New `scripts/pylyshyn/run_pylyshyn_speed_sweep.py` /
`plot_speed_sweep.py`: fixed `model=qwen/qwen3.6-plus` (no reasoning),
swept `redirect_s` (slow=2.0s / fast=1.0s) x `speed_px_s` (slow=16-33 /
fast=32-66) x variant (native/stretched), 16 seeds/condition.
qwen3.6-plus stayed strong (75-100%) across all four conditions -- no
collapse from either faster redirects or doubled speed, so the hypothesis
that yesterday's slowdown was necessary for this model didn't hold up.
Results: `results/pylyshyn/speed_sweep/accuracy_summary.png`.

## 3. Presentation GIFs

Built a one-off `scripts/prototype/mp4_to_gif.py` (cv2 + Pillow, no
ffmpeg needed) and generated 5 sample GIFs for the user's presentation: one
debug_circular clip and all four pylyshyn speed/redirect conditions from
Task 2's trial data. Copied into the repo's `gifs/` folder on request.

## 4. Trying other models on the speed/redirect sweep (Task 3, expanded well past its original scope)

TODO's Task 3 asked to try `z-ai/glm-5.3-flash` "with no reasoning" at a
bumped-up 20 seeds/condition (10+10, was 16). Live OpenRouter check found
`glm-5.3-flash` has `reasoning.mandatory=true` -- no disable option, same
situation as `qwen3.8-max` before it. Substituted `z-ai/glm-5v-turbo`
(same vendor family, reasoning optional) per the user.

New `scripts/pylyshyn/run_pylyshyn_speed_sweep_models.py` /
`plot_speed_sweep_models.py`, kept as a **separate** results/data
directory (`results/pylyshyn/speed_sweep_models/`) from Task 2's
`speed_sweep/` rather than retrofitting a model axis + variable seed count
into that script. Model is a real grid axis + part of the resume key, so
more models can be (and were) added over the course of the session:

- **`z-ai/glm-5v-turbo`**: near chance (45-70%).
- **`moonshotai/kimi-k3`**: solid, 59-95% -- but a ~35% failure rate at
  first, traced to OpenRouter falling back from a rate-limited primary
  provider to one that doesn't support video input at all. Fixed by
  pinning `provider={"only": ["moonshotai"], "allow_fallbacks": False}`.
- **`google/gemma-4-31b-it:free`**: abandoned after real effort -- its
  429s came from a rate limit pool shared across *all* OpenRouter users,
  not something client-side pacing controls. Only ~25/160 trials landed.
  Switched to the **paid** `google/gemma-4-31b-it` instead (~$0.07 for the
  full 160-trial run) -- ran cleanly, zero errors.
- **`qwen/qwen3.5-397b-a17b`**: the one model here with a real, consistent
  stretch-trick benefit (every condition shows stretched beating native,
  most dramatically 33%->80%), unlike every other pylyshyn model tested to
  date. Had a small (~4%) rate of sporadic empty-content responses
  unrelated to reasoning-budget exhaustion (reasoning was off); fixed the
  same way as the kimi-k3 429s -- clean the error rows, resume.

**Final pooled accuracy** across all 4 conditions x both variants
(`results/pylyshyn/speed_sweep_models/accuracy_summary.png`, 5 model rows):

| model | n | overall accuracy |
|---|---|---|
| qwen/qwen3.6-plus | 128 | 91.4% |
| google/gemma-4-31b-it | 160 | 91.2% |
| moonshotai/kimi-k3 | 160 | 83.8% |
| qwen/qwen3.5-397b-a17b | 160 | 60.6% |
| z-ai/glm-5v-turbo | 160 | 53.1% (chance) |

**Headline finding**: `gemma-4-31b-it`, a 31B dense model, is
statistically tied with `qwen3.6-plus` for the best score on this
stimulus, and clearly beats much larger models (`kimi-k3`'s a 2.8T-param
MoE per its own listing; `qwen3.5-397b-a17b` is 397B total params). Raw
parameter count doesn't predict pylyshyn performance here -- scoped
strictly to this stimulus, not a general video-understanding claim.

## 5. Following up on the gemma surprise: debug_circular angle sweep (Task 4)

Before running, disambiguated *why* `gemma-4-31b-it`'s tokens don't respond
to the duration-stretch trick: varying the *actual* rendered frame count
(100/200/400 real frames, native encoding) produced exactly identical
`prompt_tokens` every time -- it extracts a fixed-size frame sample
regardless of real video length, not "reads every frame, robust to
tricks." Written up in `reference_gemma_fixed_frame_budget`.

New `scripts/debug_circular/run_debug_circular_angle_sweep.py` /
`plot_angle_sweep.py`: `gemma-4-31b-it` across `rotation_deg` in
{40,...,360}, native vs. stretched, 20 seeds/condition (360 trials, 0
errors). As expected, native and stretched tracked each other almost
perfectly. Unexpectedly, the accuracy curve was **V-shaped, not flat** --
near-ceiling at the extremes (40/360 deg, ~90-95%) but *below chance* in
the middle (200-240 deg, ~10-15%).

Hypothesized this was a position-heuristic effect: `n_objects=3` spaced
120 deg apart means a rotation near a multiple of 120 deg moves the cued
cross into another cross's old slot, which would fool position-based (not
identity-based) tracking. Tested this directly by generalizing the script
(`--run-name`/`--n-objects`/`--rotation-degs`/`--variants`, plus fixing its
resume key to include `n_objects`) and re-running at `n_objects=4` (90 deg
spacing) across 8 angles alternating on/off the new symmetry points
(160 trials, 0 errors, ~$0.04). **Result: hypothesis not confirmed** -- if
it were true, accuracy should recover at the off-symmetry angles (135/225
deg) between the 90/180/270 danger points. Instead the curve stayed
smoothly, continuously low across the whole 90-270 deg range, mirroring
n=3's curve bottoming out near (but not precisely at) its own symmetry
points. The more accurate read: a continuous rotation-magnitude bias,
hardest to judge near 180 deg and easiest near a full 0/360 deg loop
closure, independent of `n_objects`'s spacing -- not object-count-driven
position confusion. Recorded as tested-and-not-confirmed rather than
forced to fit the original guess.

## Process gotchas worth remembering

- **Seed-subset smoke tests corrupt the matching/non-matching lock**:
  `run_pylyshyn_stretch_sweep.py`/`run_pylyshyn_speed_sweep.py` derive
  `probe_on_target` from each seed's rank within the full `--seeds` list,
  locked via `config.json`. Smoke-testing with a smaller seed subset than
  the intended real run reassigns some seeds' matching/non-matching label,
  and since the resume key includes `probe_on_target`, the mismatch isn't
  recognized as "already run" -- it silently appends a stale duplicate row
  instead of erroring. Caught twice today by eyeballing an `n` that didn't
  match the expected seed count.
- **Three distinct OpenRouter provider-routing gotchas**, written up in
  full in memory (`reference_openrouter_provider_routing_gotchas`):
  fallback to a video-incapable provider, `:free` models sharing a
  cross-user rate-limit pool, and sporadic empty-content responses that
  look like the known reasoning-budget-exhaustion bug but aren't (check
  `reasoning_tokens` first).

## Not yet done

- `google/gemma-4-31b-it:free`'s partial data (~25/160 rows) sits in
  `results/pylyshyn/speed_sweep_models/results.csv` for reference but
  isn't in the comparison plot -- not worth forcing given the shared-pool
  rate limit, but could be revisited if it ever loosens up.
- The `gemma-4-31b-it` vs. much-larger-models finding is still open beyond
  today's debug_circular follow-up -- worth a deliberate pass (more
  small/dense models, or probing what `gemma-4-31b-it` and `qwen3.6-plus`
  share) rather than leaving it as a one-off surprise.
- The debug_circular V-shaped rotation-angle curve is real and replicated
  (n_objects=3 and 4), but its actual mechanism is still unexplained now
  that the position-heuristic theory is ruled out -- a finer-grained angle
  sweep or a direct rotation-magnitude-estimation task (instead of the
  binary same/different judgment used here) would test whether this is a
  genuine perception limit around 180 deg specifically.
- `smooth_pursuit` arm still untouched since 2026-08-14.
- `n_objects=3` still likely too easy for the strongest configs now
  (several models near ceiling); scaling field size remains the natural
  next lever.
