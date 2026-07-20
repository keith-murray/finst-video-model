# Video Generation Model FINST/MOT Capacity Experiment

Tests whether video generation models show a Pylyshyn-style capacity limit
when tracking a cued subset of identical circles through a de-cued
"invisible tracking" phase (adapted from classic multiple-object-tracking /
MOT paradigms). All models are accessed through OpenRouter, so in principle
any video generation model OpenRouter exposes can be tested this way — Veo
3.1 is simply the first model this has been tried on.

## Design summary

Video generation models typically accept a single starting **image** (no
multi-frame video conditioning for arbitrary/external video), so the full
three-phase MOT trial structure (cue -> de-cue/track -> re-cue) is encoded
entirely in the **text prompt**, anchored to a single ground-truth frame:

1. **Frame 0** (rendered by us, `stimulus_gen.py`): N identical circles,
   K of them colored red (cued), rest gray. This is the only visual
   ground truth we control.
2. **Prompt** (owned by each experiment script under `scripts/`, e.g.
   `run_trial.py` / `run_circular_trial.py`): instructs the model to (a)
   de-cue within ~1s so all circles become identical gray, (b) move all
   circles for several seconds under some described motion, (c) re-cue
   *only the originally-red circles* in the final frame.
3. The model generates the whole clip from image + prompt in one call.

**Important limitation to keep in mind**: these models are generative, not
physics simulators. They will not execute the described motion exactly.
This means there is no external ground-truth trajectory to check the
output against — verification must be done against what the model *itself*
depicted (see "Verification pipeline" below), not against an intended
physics simulation.

## What's implemented

- `config.py` — `TrialConfig` dataclass: all independent variables
  (n_circles, n_cued, phase durations, geometry, colors, seed, plus
  circular-track geometry). Validates phase durations sum to a supported
  clip length (Veo 3.1 supports 4/6/8s; other models may differ).
- `stimulus_gen.py` — renders frame 0 via PIL. Two stimulus types:
  `generate_stimulus` (rejection-sampled non-overlapping circle placement,
  free-form motion) and `generate_circular_track_stimulus` (circles evenly
  spaced around a drawn circular track, for constrained clockwise motion —
  this constraint noticeably reduced hallucinated motion compared to
  free-form physics description). Both write `frame0.png` plus
  `ground_truth.json` into a trial-specific output directory.
- `client.py` — generic OpenRouter submit/poll/download client. Knows
  nothing about any specific experiment's stimulus, prompt, or model
  choice; `run_trial` takes an already-built prompt, frame-0 image path,
  and an OpenRouter model slug (e.g. `"google/veo-3.1"`), and launches it.
- `scripts/run_trial.py` / `scripts/run_circular_trial.py` — one script per
  experiment, each owning its own prompt text (built from a `TrialConfig`)
  and model choice (a `MODEL` constant), alongside the code that generates
  its stimulus and launches the trial via `client.run_trial`. Prompts and
  model choice intentionally live next to the launch code rather than in
  the shared package, so both can be iterated on per-experiment without
  touching shared infra.

Every trial's artifacts (`frame0.png`, `ground_truth.json`, `video.mp4`,
`generation.json`) land in `data/<trial_id>/`, linked by `trial_id`.

## What's NOT implemented yet (next steps for Claude Code)

### 1. Frame extraction
- Use `ffmpeg` or `opencv` to pull all frames (or at minimum the last ~1s
  worth of frames) from the generated video.

### 2. Verification / blob-tracking pipeline
This is the core scoring logic and needs the most design care:
- **Blob detection**: per frame, detect circular blobs by color
  (red vs. gray vs. background) via simple color thresholding + contour
  detection (cv2.findContours or similar) — should be robust given the
  deliberately simple, flat-color stimulus design.
- **Frame-to-frame tracking**: nearest-centroid tracking across frames to
  build each blob's trajectory through the *generated* video (this is
  necessary since these models don't give you object IDs — you have to
  reconstruct them from pixel output, the same way you'd analyze a real
  MOT trial video).
- **Validity/exclusion checks** (run before scoring):
  - Exactly N circles detected in frame 0 of the *output* (sanity check
    that the model rendered the input image correctly).
  - Circle count stays constant throughout (no unexplained merges/splits/
    disappearances) — flag or exclude trials that fail this.
  - Exactly K circles are red in the final frame — trials with the wrong
    count should be scored as failures on their own (this is the "cheap
    failure mode" check), separate from identity-correctness below.
- **Identity scoring**: for each circle red in the final frame, walk its
  tracked trajectory backward to frame 0 of the *generated* video and check
  whether it corresponds to a circle that was red in frame 0. Compute
  accuracy as fraction of correctly-identified cued circles per trial.

### 3. Batch runner
- Sweep `n_circles` across e.g. [2, 3, 4, 5, 6, 8, 10, 12], fixed
  `n_cued=1` or `2`, multiple seeds/repeats per N (for noise — generation
  is stochastic).
- Log per-trial: config, prompt, model, ground_truth, output video path,
  extracted trajectories, validity flags, accuracy score.
- Output a tidy CSV/DataFrame for downstream analysis (accuracy vs. N plot
  is the headline result — looking for a capacity "knee" around N=4-5 per
  Pylyshyn/MOT literature).

## Comprehension arm (video-understanding VLM)

Motivated by video-generation models' own documented weakness at multi-object
motion fidelity (the Veo arm above hit this directly — see
`claude/2026_07_18/TODO.md` results), this arm sidesteps generation entirely:
we render the video ourselves with deterministic physics (exact ground truth,
nothing to reverse-engineer from output pixels), and ask a video-*understanding*
VLM a text question instead of asking a video-*generation* model to move
objects correctly.

- `src/finst_video_model/comprehension/config.py` — a separate `TrialConfig`
  (distinct from the Veo arm's) with `fps`/`tracking_s`/`speed_px_s` physics
  timing instead of a fixed clip duration, plus `force_path_crossing` (biases
  cued circles' initial headings toward the distractor centroid, to increase
  trajectory-crossing events during tracking) and `use_label_phase` — a
  hyperparameter for whether the video ends with a frozen, letter-labeled
  frame. Label phase is needed for the VLM text-question task below; disable
  it for uses that only need the raw cue -> tracking motion (e.g. probing a
  frozen encoder's activations directly against ground truth, no text answer
  involved).
- `comprehension/physics.py` — deterministic constant-velocity motion with
  elastic wall bounces; non-overlapping initial placement via rejection
  sampling.
- `comprehension/stimulus_gen.py` — renders the full mp4 (via OpenCV
  `VideoWriter`) across the cue / tracking / (optional) label phases, and
  writes `ground_truth.json` with every circle's final position, assigned
  letter (`null` if `use_label_phase=False`), and cued/not-cued status, plus
  the answer key (`cued_letters`, `null` if no label phase). Same
  `data/<trial_id>/` artifact convention as the Veo arm.
- `scripts/run_comprehension_trial.py` — the one script that owns this arm's
  question text (`build_question`, matching the convention that prompts live
  next to launch code, not in the shared package) and runs stimulus
  generation end to end, writing `video.mp4` / `ground_truth.json` /
  `question.txt` into `data/<trial_id>/`.
- `scripts/run_extend_trial.py` — a sibling script for a different model
  class: instead of asking a VLM a text question, this renders the same
  cue -> tracking motion with `use_label_phase=False` (no frozen/labeled
  ending) and writes an `extend_prompt.txt` asking a video-*extension* model
  (one that continues an existing video from its last frame) to bring the
  circles to a stop and recolor only the originally-cued ones red — closer
  to the original Veo generation-arm task, but handing off only the re-cue
  step instead of the whole clip. No model call is wired up yet (OpenRouter
  video-extend support is still beta/sparse per
  `claude/2026_07_20/TODO.md`) — this only produces the base video + prompt.

**What's NOT implemented yet for this arm**:

- No model call yet — the video and question are generated but nothing
  sends them to a VLM. Needs a thin adapter per target API (Gemini,
  Qwen3-VL, GPT-5, etc.) that uploads/attaches the video, sends the
  question, and returns raw text.
- Answer parsing (letters out of free text) and scoring against
  `ground_truth["cued_letters"]` (exact-set match, plus partial-credit
  precision/recall).
- Batch runner sweeping `n_circles` x `force_path_crossing` x seeds.

## Open design questions to resolve before scaling up

- Whether to sweep `n_cued` (K) as a second axis, not just `n_circles`.
- Whether to also vary "path crossing" (deliberately prompting for cued vs.
  distractor paths to cross) as a separate manipulation, per the
  feature-swap/tunnel-effect logic discussed earlier in this project.
- How many repeats per condition are needed given generation cost/latency
  — worth a small pilot (e.g., n=5 per condition) before committing to a
  full sweep.
- Whether/how prompt wording and request payload fields (duration, size,
  audio) need to vary across models, now that `client.py` no longer
  hardcodes Veo — different OpenRouter video models may expect different
  parameters or tolerate different phrasing.
