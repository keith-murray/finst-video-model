# Veo 3.1 FINST/MOT Capacity Experiment

Tests whether Veo 3.1 shows a Pylyshyn-style capacity limit when tracking a
cued subset of identical circles through a de-cued "invisible tracking"
phase (adapted from classic multiple-object-tracking / MOT paradigms).

## Design summary

Veo 3.1 only accepts a single starting **image** (no multi-frame video
conditioning for arbitrary/external video — see note below), so the full
three-phase MOT trial structure (cue -> de-cue/track -> re-cue) is encoded
entirely in the **text prompt**, anchored to a single ground-truth frame:

1. **Frame 0** (rendered by us, `stimulus_gen.py`): N identical circles,
   K of them colored red (cued), rest gray. This is the only visual
   ground truth we control.
2. **Prompt** (`prompts.py`): instructs Veo to (a) de-cue within ~1s so all
   circles become identical gray, (b) move all circles for several seconds
   under described physics, (c) re-cue *only the originally-red circles* in
   the final frame.
3. Veo generates the whole 8s clip from image + prompt in one call.

**Important limitation to keep in mind**: Veo is generative, not a physics
simulator. It will not execute the described motion exactly. This means
there is no external ground-truth trajectory to check the output against —
verification must be done against what Veo *itself* depicted (see
"Verification pipeline" below), not against an intended physics simulation.

## What's implemented

- `config.py` — `TrialConfig` dataclass: all independent variables
  (n_circles, n_cued, phase durations, geometry, colors, seed). Validates
  phase durations sum to a Veo-supported clip length (4/6/8s).
- `stimulus_gen.py` — renders frame 0 via PIL (rejection-sampled
  non-overlapping circle placement), writes the PNG plus a
  `ground_truth_<trial_id>.json` recording every circle's position and
  cued/not-cued status.
- `prompts.py` — builds the three-phase text prompt from a `TrialConfig`.

Both have smoke tests under `if __name__ == "__main__"` and have been run
successfully (see `stimuli/` for a sample output).

## What's NOT implemented yet (next steps for Claude Code)

### 1. Veo API call wrapper
- Use `client.models.generate_videos(model="veo-3.1-generate-preview", prompt=..., image=..., config=types.GenerateVideosConfig(...))`.
- Must poll `operation.done` (see Gemini API video generation docs) and
  download the result.
- `resolution="720p"`, `duration_seconds` should equal `cfg.clip_duration_s`
  as a string (`"4"`, `"6"`, or `"8"`).
- Handle safety-filter blocks/failures gracefully — log and skip rather than
  crash a batch run (circles-on-plain-background prompts *shouldn't* trip
  filters, but audio-safety blocks have been reported as a general Veo 3.1
  failure mode; also handle empty/malformed responses).
- Note API cost/latency: each 8s 720p generation can take up to several
  minutes (documented range: 11s min, up to 6 min at peak). Batch runs
  over multiple N values x multiple repeats will need async/queued
  execution, not a blocking loop, to be practical.
- Save output video alongside the trial's `ground_truth_<trial_id>.json`
  (same trial_id) so config, ground truth, and output stay linked.

### 2. Frame extraction
- Use `ffmpeg` or `opencv` to pull all frames (or at minimum the last ~1s
  worth of frames) from the generated 24fps video.

### 3. Verification / blob-tracking pipeline
This is the core scoring logic and needs the most design care:
- **Blob detection**: per frame, detect circular blobs by color
  (red vs. gray vs. background) via simple color thresholding + contour
  detection (cv2.findContours or similar) — should be robust given the
  deliberately simple, flat-color stimulus design.
- **Frame-to-frame tracking**: nearest-centroid tracking across frames to
  build each blob's trajectory through the *generated* video (this is
  necessary since Veo doesn't give you object IDs — you have to reconstruct
  them from its pixel output, the same way you'd analyze a real MOT trial
  video).
- **Validity/exclusion checks** (run before scoring):
  - Exactly N circles detected in frame 0 of the *output* (sanity check
    that Veo rendered the input image correctly).
  - Circle count stays constant throughout (no unexplained merges/splits/
    disappearances) — flag or exclude trials that fail this.
  - Exactly K circles are red in the final frame — trials with the wrong
    count should be scored as failures on their own (this is the "cheap
    failure mode" check), separate from identity-correctness below.
- **Identity scoring**: for each circle red in the final frame, walk its
  tracked trajectory backward to frame 0 of the *generated* video and check
  whether it corresponds to a circle that was red in frame 0. Compute
  accuracy as fraction of correctly-identified cued circles per trial.

### 4. Batch runner
- Sweep `n_circles` across e.g. [3, 4, 5, 6, 8, 10, 12], fixed `n_cued=2`,
  multiple seeds/repeats per N (for noise — Veo generation is stochastic).
- Log per-trial: config, prompt, ground_truth, output video path, extracted
  trajectories, validity flags, accuracy score.
- Output a tidy CSV/DataFrame for downstream analysis (accuracy vs. N plot
  is the headline result — looking for a capacity "knee" around N=4-5 per
  Pylyshyn/MOT literature).

## Open design questions to resolve before scaling up

- Whether to sweep `n_cued` (K) as a second axis, not just `n_circles`.
- Whether to also vary "path crossing" (deliberately prompting for cued vs.
  distractor paths to cross) as a separate manipulation, per the
  feature-swap/tunnel-effect logic discussed earlier in this project.
- How many repeats per condition are needed given Veo's generation cost/
  latency — worth a small pilot (e.g., n=5 per condition) before committing
  to a full sweep.
