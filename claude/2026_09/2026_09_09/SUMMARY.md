# Summary: prepared a local-gemma-4-31b-it frame-count sweep on debug_circular

Full context/goals: `claude/2026_09/2026_09_09/TODO.md`. This machine has no
SSH access to the cluster (`scotty` -- confirmed via a failed key-auth
attempt), so this session only prepares stimuli + scripts locally; the actual
`sbatch` run needs to happen from the cluster (the user, or a separate
on-cluster Claude session, per the existing `qwen38_cluster_handoff.md`
handoff pattern).

## What was built

1. **40 stimuli generated locally**
   (`data/debug_circular/gemma_frame_sweep/trials/`, gitignored) via the
   existing `scripts/debug_circular/generate_samples.py --rotation-deg 40
   --seeds-per-condition 10 --no-save-mp4` -- rotation_deg=40 is the angle
   OpenRouter's hosted gemma-4-31b-it scored ~93% native accuracy on
   (`project-status-2026-08-27`). Verified: exactly 20 `probe_is_target=True`
   / 20 `False`, each `video.npy` is `(100, 384, 384, 3)` uint8 RGB, no mp4s
   written.

2. **`scripts/debug_circular/run_gemma_cluster_batch.py`** -- new cluster
   batch script, self-contained (runs in `$HOME/local-llm/.venv`, no
   `finst_video_model` import), modeled on the qwen3.8-27b pipeline's
   `run_cluster_batch.py` but built on `transformers.AutoProcessor`/
   `AutoModelForMultimodalLM` (confirmed-working pattern from
   `claude/skills/cluster/gemma4/test_gemma4_baseline.py`) instead of vLLM.
   Loops over every `(trial_id, num_frames, sampling)` triple, controlling the
   frame budget via `do_sample_frames=True, num_frames=N` on
   `processor.apply_chat_template()`. Processes triples **serially** (one
   `model.generate()` call at a time) rather than batched -- there's no
   confirmed multi-video batching pattern for this model/processor, and
   introducing one untested felt riskier than the extra wall-clock cost.
   Writes `cluster_response_nframes{N}_{sampling}.json` per triple, resumable
   (skips triples whose response file already exists).

   **Sampling axis added mid-session** (user follow-up question caught a real
   gap): neither `test_gemma4_baseline.py` nor this script's first draft set
   any generation sampling params at all -- `model.generate()` was only ever
   called with `max_new_tokens`, silently falling back to whatever
   `generation_config.json` ships with the checkpoint (unverified from this
   machine). The qwen3.8-27b pipeline, by contrast, explicitly pins
   `temperature=0.0, top_p=1.0` for determinism. Fixed by adding a
   `SAMPLING_CONFIGS` dict with two explicit modes swept together: `"greedy"`
   (`do_sample=False`, deterministic) and `"recommended"`
   (`do_sample=True, temperature=1.0, top_p=0.95, top_k=64` -- HuggingFace's
   documented defaults for this model, per the user). `"recommended"` is
   stochastic and each trial only gets one draw (not averaged over repeated
   samples), so its per-condition accuracy is noisier than `"greedy"`'s --
   worth keeping in mind when reading the eventual results.

3. **`scripts/debug_circular/profile_gemma_cluster_timing.py`** -- timing
   probe (2 sample trials x all 7 `num_frames` values x both sampling modes
   by default) to measure real per-call generation time before committing to
   a `--time` budget for the full 40 x 7 x 2 = 560-call sweep, mirroring the
   qwen pipeline's `profile_cluster_timing.py` convention of measuring rather
   than guessing. Also a chance to eyeball whether `"recommended"`'s
   stochastic output still looks like a sane True/False answer.

4. **`slurm/run_gemma_frame_sweep.sh`** -- sbatch script, 2x A100-40G (same
   as `test_gemma4_baseline.sh`, sized for gemma's ~62GB memory footprint
   across 2 GPUs), `--time` left as an explicit placeholder with a comment
   pointing at the profiling script's output.

5. **`scripts/debug_circular/aggregate_gemma_frame_sweep.py`** -- reads all
   `cluster_response_nframes{N}_{sampling}.json` files + `ground_truth.json`,
   reuses `finst_video_model.scoring` (`parse_boolean_answer`/`classify_trial`/
   `compute_d_prime`) exactly like `aggregate_cluster_results.py`, writes
   `results/debug_circular/gemma_frame_sweep/results.csv` with `num_frames`
   and `sampling` columns and prints d' grouped by `(num_frames, sampling)`.
   Kept as a separate script rather than generalizing the existing
   rotation_deg aggregator, since these sweep axes are properties of the
   inference run, not of `TrialConfig`/`ground_truth.json`.

6. **`scripts/debug_circular/plot_gemma_frame_sweep.py`** -- line + SEM-
   errorbar plot (styling/color constants copied from `plot_angle_sweep.py`'s
   two-line native/stretched convention for visual consistency), x-axis =
   `num_frames`, one line per sampling mode, with a vertical dotted reference
   line at `num_frames=32` labeled "OpenRouter cap" so the motivating
   comparison (does accuracy near/above the cap differ from below it, does
   feeding more than 32 real frames -- impossible on OpenRouter -- help) is
   visible directly on the figure.

**Verified end-to-end locally with synthetic fake response data** (not
committed, scratch-only) that `aggregate_gemma_frame_sweep.py` ->
`plot_gemma_frame_sweep.py` produces a correctly-styled, correctly-scored
figure before ever touching the cluster -- caught no bugs on either the
original single-sampling-mode version or after adding the sampling axis, both
ran cleanly on the first try each time.

## Real cluster runs landed today, and both raised the central open question

Two real sweeps ran on the cluster today (a third parallel on-cluster Claude
session drove the actual `sbatch` submissions and committed results directly
into the repo -- this session prepared the scripts, then diagnosed the
results as they came back). Both came back with accuracy far below
OpenRouter's ~93% baseline at `rotation_deg=40`, and neither of the two
leading hypotheses tested today explains the gap:

### Run 1: `do_sample_frames=True` auto-sampling (`results/debug_circular/gemma_frame_sweep/`)

First real run predated the sampling-mode fix (its `results.csv`/`config.json`
still use the old single-condition schema), so it used whatever unspecified
`generate()` defaults the checkpoint ships with. Raw accuracy
(`accuracy_by_n_frames.png`) peaked at 65-68% around `num_frames=16-24` and
sat at ~50% (chance) at both `num_frames=4-8` and `num_frames=100`.

**Diagnosed directly from `results.csv`'s `predicted` column, not just the
plot: this is a response-bias artifact, not a real sensitivity collapse.**
The model gives one fixed answer almost regardless of the actual video, and
which fixed answer flips with frame count:

| num_frames | predicted True/False (of 40) | raw accuracy | d' | criterion |
|---|---|---|---|---|
| 4 | 40/0 | 50% | **0.00** | -1.98 (always "True") |
| 8 | 40/0 | 50% | **0.00** | -1.98 (always "True") |
| 16 | 20/20 | 65% | 0.73 | 0.00 |
| 24 | 7/33 | 68% | **1.61 (peak)** | +1.17 |
| 32 | 3/37 | 57% | 1.01 | +1.47 |
| 50 | 1/39 | 52% | 0.52 | +1.72 |
| 100 | 2/38 | 50% | **0.00** | +1.47 (always "False") |

d'=0 at both extremes reflects a fixed default answer, not random guessing --
same failure signature as `project-status-2026-08-26`'s qwen3.8-27b-nothink
"always False, d'=0 chance floor" finding. Real sensitivity peaks in the
middle (d'=1.61 at `num_frames=24`, a genuine inverted-U) but still falls
well short of OpenRouter's implied ~d'=3+.

### Sampling-mode fix, then a proper 2-condition rerun: sampling ruled out

Added `SAMPLING_CONFIGS` to `run_gemma_cluster_batch.py` (`"greedy"`:
`do_sample=False`; `"recommended"`: `do_sample=True, temperature=1.0,
top_p=0.95, top_k=64` -- HF's documented defaults for this model, confirmed
against the model card by the user). The full 40 x 7 x 2 = 560-call rerun
came back with **greedy and recommended nearly identical** -- at
`num_frames` 4, 24, and 32 the prediction counts, d', and criterion are
*exactly* the same; only 16/50/100 show a couple of flipped predictions,
consistent with sampling noise around an already near-deterministic decision.
**Sampling mode is not the cause.** Also directly ruled out: prompt wording
(`run_gemma_cluster_batch.py`'s `build_question()` is byte-for-byte identical
to `run_openrouter_diagnostic.py`'s, which produced the OpenRouter baseline)
and reasoning/thinking (confirmed both via code --
`run_debug_circular_angle_sweep.py` hardcodes `REASONING = {"enabled":
False}` -- and via data -- all 360 rows of the original OpenRouter
`angle_sweep/results.csv` show `reasoning_tokens=0`).

The most damning single data point: **`num_frames=100` is the full,
untruncated native clip -- no subsampling possible -- and it still only
reaches d'≈0.5-1.0**, far below what OpenRouter's ~93% implies despite
OpenRouter only ever seeing a 32-frame subsample of that same clip. Full
local information underperforming heavily-subsampled cloud information means
the frame *budget* was never the whole story.

### Run 2: bypass `do_sample_frames` with manual frame selection -- also ruled out

Leading hypothesis going in: `do_sample_frames=True`'s auto-sampler might be
silently mis-selecting frames the same way vLLM's `video_url` connector did
for the qwen3.8-27b pipeline (`qwen38_cluster_handoff.md` Known Bug #5 --
direct precedent in this repo for exactly this failure class). Built
`scripts/debug_circular/run_gemma_cluster_batch_manual_frames.py` (kept
separate from `run_gemma_cluster_batch.py` per the user's request) that
computes N evenly-spaced frame indices itself via `np.linspace` and passes
`do_sample_frames=False` -- verified locally (pure numpy, no cluster needed)
that this guarantees frame 0 (cue) and frame 99 (probe) are *always* included
in the sample, even at `num_frames=4`. Also added `--response-prefix
{nframes,manualframes}` to `aggregate_gemma_frame_sweep.py` so both
pipelines' output can be scored without colliding.

**Result (`results/debug_circular/gemma_frame_sweep_manualframes/`): the
exact same bias signature persists.** Guaranteeing the probe frame is seen
did not fix the `num_frames=4` "always True" collapse, nor the
`num_frames=50/100` "always False" collapse:

| num_frames | manual-slice d' (greedy) | auto-sample d' (Run 1, greedy) |
|---|---|---|
| 4 | 0.00 (always "True") | 0.00 (always "True") |
| 16 | **1.47** | 0.64 |
| 24 | 1.19 | **1.61 (Run 1's peak)** |
| 32 | 1.34 | 1.01 |
| 100 | 0.52 | 0.00 (always "False") |

Peak d' (~1.2-1.6) and overall shape are comparable between the two frame-
selection strategies -- **this rules out `do_sample_frames` mis-sampling as
the explanation.** Sampling mode, prompt wording, reasoning, and frame-
selection mechanism are now all ruled out.

## Next directions to close the gap (deferred to next session, per the user)

With the sweep-parameter space exhausted, the remaining candidates are about
the pipeline itself differing from OpenRouter's, not about what's being
swept:

1. **Image/video preprocessing mismatch** (leading candidate) -- when
   `AutoProcessor` receives a raw numpy array directly (this pipeline's path)
   vs. decoding an actual `.mp4` (OpenRouter's path), it may apply different
   default resize/crop/normalization, potentially degrading effective
   resolution enough to blur the crosses' fine positional/color detail.
   Cheap to check: print `processor.image_processor`'s actual config
   (resize/crop settings) on the cluster -- no GPU generation needed.
2. **Checkpoint identity** -- confirm `/scratch/km3199/models/gemma-4-31B-it`
   is bit-for-bit the same instruction-tuned weights OpenRouter serves, not a
   different quantization or a community re-upload with a subtly different
   vision tower. Check the checkpoint's own `config.json`/revision info.
3. **Qualitative sanity check** -- swap the True/False question for "describe
   this video in detail" on a couple of the best-performing trials
   (`num_frames=16`, the current peak) to see whether the model's basic
   color/position/motion perception is intact at all, before continuing to
   assume the deficit is specifically in the identity-tracking judgment
   rather than in raw visual perception.
4. Path ambiguity across gemma-cluster files is still unresolved but appears
   moot in practice -- the on-cluster session's runs completed successfully
   using whichever path it actually resolved to, so this is lower priority
   than 1-3 now.
