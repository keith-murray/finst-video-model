# VLM FINST/MOT Capacity Experiment (comprehension arm)

Tests whether a video-understanding VLM shows a Pylyshyn-style capacity
limit when asked to report which of N identical circles were briefly cued
at the start of a video, after a de-cued "invisible tracking" phase.

This is the comprehension-task counterpart to the Veo generation-task arm:
same underlying MOT logic (cue -> de-cue/track -> report), but here the
model only has to answer in **text**, which removes the biggest problem we
hit with Veo -- there is no need to reverse-engineer ground truth from
generated pixels, because we fully simulate and render the video ourselves.

## Two-stage reality check (read this before choosing a target model)

Meta has released the raw V-JEPA2 encoder weights (loadable via
`transformers`, e.g. `facebook/vjepa2-vitg-fpc64-384-ssv2`), but the
paper's VidQA MLLM -- V-JEPA2 encoder + projector + Llama 3.1 8B / Qwen2-7B,
aligned together for open-ended video question answering -- does not
appear to have its trained projector/alignment weights publicly released.
There is no single off-the-shelf "V-JEPA2 chatbot" endpoint to call.

This splits the project into two stages:

- **Stage 1 (this pipeline is ready for it now)**: run the behavioral task
  against any existing callable video-understanding VLM (Gemini, Qwen3-VL,
  GPT-5, InternVL, etc.) to get a first, immediate read on whether a
  capacity limit shows up behaviorally at all, in any current model.
- **Stage 2 (bigger lift, not started)**: to get the *mechanistic* access
  V-JEPA2 specifically motivated (probing whether the encoder's own
  activations carry a persistent object index, per our earlier FINST
  discussion), you'd need to either (a) assemble your own LLaVA-style MLLM
  by training a projector between the open V-JEPA2 encoder and an open LLM
  yourself, or (b) search for a community-trained checkpoint that does
  this pairing, or (c) skip the LLM entirely and probe the frozen V-JEPA2
  encoder's activations directly against ground-truth identity labels
  (no text answer needed for this -- just linear probing, closer to what
  we discussed for the object-centric-model arm).

Recommendation: run Stage 1 first against 2-3 off-the-shelf VLMs to
establish whether the behavioral effect exists anywhere before committing
engineering time to Stage 2.

## Design summary

Because we render the video ourselves (unlike the Veo arm), we have exact,
authored ground truth for every circle at every frame -- no CV/tracking
pipeline needed to figure out what "actually happened."

1. **Cue phase** (`cue_flash_s`, default 1.0s): K of N circles are red, rest
   gray, all already moving.
2. **Tracking phase** (`tracking_s`, default 6-8s): all circles turn and
   stay identical gray, continue moving under simple constant-velocity
   motion with elastic wall bounces (see `physics.py`). No circle-circle
   collision physics -- circles may visually cross paths, which is
   intentional (see `force_path_crossing` below), not a bug.
3. **Label phase** (`label_s`, default 2.0s): motion freezes at its final
   position, and every circle gets a single letter (A, B, C, ...) overlaid,
   in a randomized letter-to-circle assignment (recorded in ground truth,
   not derivable from position). Held for the last `label_s` seconds so
   the VLM's frame sampler reliably catches at least one labeled frame.

The model is asked, in text, which letter(s) correspond to the circles
that were red at the start. Scoring is a simple set-comparison between the
model's answer and `ground_truth["cued_letters"]` -- no vision pipeline
required on the output side.

## What's implemented

- `config.py` -- `TrialConfig` dataclass: n_circles, n_cued, phase timings,
  speed, geometry, colors, seed, and a `force_path_crossing` flag (biases
  cued circles' initial headings toward the distractor centroid, to
  increase trajectory-crossing events during tracking -- this is the
  feature-swap/tunnel-effect-style manipulation from our earlier
  discussion, adapted for this design).
- `physics.py` -- deterministic constant-velocity motion with elastic wall
  bounces; non-overlapping initial placement via rejection sampling.
- `stimulus_gen.py` -- renders the full mp4 (via OpenCV VideoWriter) across
  all three phases, and writes `ground_truth_<trial_id>.json` with every
  circle's final position, assigned letter, and cued/not-cued status, plus
  the answer key (`cued_letters`).
- `prompts.py` -- builds the text question sent alongside the video.

All three have been smoke-tested (`stimuli/` contains a sample run:
6 circles, 2 cued, cued letters `['C', 'E']` for that seed) and frames from
each phase were visually spot-checked (cue-phase red circles visible,
tracking-phase all-gray, label-phase letters correctly overlaid).

## What's NOT implemented yet (next steps for Claude Code)

### 1. VLM API call wrapper(s) -- Stage 1
- Needs a thin adapter per target API (Gemini, Qwen3-VL, GPT-5, etc.) that:
  uploads/attaches `video_path`, sends `build_question(cfg)` as the text
  prompt, and returns the raw text response.
- Check each API's actual frame-sampling rate / max frame count against
  `cfg.total_duration_s` and `cfg.fps` before running -- if a model's
  native sampling is sparse (e.g. 1fps), consider whether `tracking_s`
  needs to be shortened or the video needs to be slower/simpler so the
  cue flash and de-cue transition aren't sampled away entirely.
- Parse the model's free-text response into a set of letters (simple
  regex for capital letters should suffice given the constrained answer
  format we asked for, but validate against a few real responses first --
  models may add hedging text despite instructions).

### 2. Scoring
- Compare parsed answer letters against `ground_truth["cued_letters"]`
  (exact set match). Worth recording partial-credit metrics too (e.g.
  precision/recall over letters) in case models get the right count but
  wrong identity, or vice versa -- that distinction is exactly the
  Pylyshyn-relevant signal, same as it was for the Veo arm.
- Also record: did the model return exactly `n_cued` letters at all
  (count-only correctness, the "cheap" success mode) separately from
  full identity correctness.

### 3. Batch runner
- Sweep `n_circles` (e.g. [3,4,5,6,8,10,12]) x `force_path_crossing`
  (True/False) x multiple seeds, fixed `n_cued=2`. Log config, question,
  ground truth, raw model response, parsed answer, and all scoring metrics
  to a tidy dataframe/CSV per model tested.

### 4. Stage 2 scaffolding (only after Stage 1 results justify it)
- If pursuing the custom V-JEPA2 MLLM: minimal path is loading a frozen
  `facebook/vjepa2-*` encoder via `transformers`, extracting per-tubelet
  embeddings for each trial video, and training a linear probe to decode
  `circle.cued` from the frozen features at various points in the
  tracking phase -- this alone (no LLM needed) already tests the
  perceptual-index hypothesis from our earlier discussion, and is a much
  smaller lift than building the full MLLM.

## Open design questions

- Whether `tracking_s` should scale with `n_circles` (more objects may
  need more time to produce meaningful crossing events) or stay fixed.
- How many repeats per condition, and whether determinism (same seed =
  same trial) makes repeats meaningless for a given model unless sampling
  temperature > 0 -- worth deciding per-model.
- Whether letter labels themselves introduce a confound (e.g. models may
  have priors about "A" being a more salient/default answer) -- worth a
  sanity check comparing accuracy across which letter ends up assigned to
  the cued circles.
