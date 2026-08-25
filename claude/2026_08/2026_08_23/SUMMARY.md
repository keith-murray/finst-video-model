# Summary: save smooth-pursuit stimuli as numpy arrays

Full context/goals: `claude/2026_08_23/TODO.md` -- investigating whether mp4
encoding/decoding is introducing artifacts that contribute to poor Qwen3.8
performance on the smooth-pursuit and pylyshyn tasks, ahead of locally
hosting `qwen3.8-27b` (via vLLM) to control video-sampling fps directly
rather than relying on OpenRouter's server-side handling. First step: get a
raw-pixel baseline to compare the mp4 against.

## 1. `save_npy` flag added to the smooth-pursuit stimulus pipeline

Scoped to the smooth-pursuit arm only (not pylyshyn) and to
`scripts/smooth_pursuit/generate_samples.py` only (not the real
OpenRouter-hitting `run_smooth_pursuit_trial.py`/`run_smooth_pursuit_batch.py`),
per explicit scope decisions during planning.

- `comprehension/smooth_pursuit/stimulus_gen.py`'s `generate_stimulus` gained
  a `save_npy: bool = False` parameter. When set, every frame handed to
  `cv2.VideoWriter.write(...)` (cue phase, tracking phase, and the
  repeated-frame probe phase) is also converted from cv2's native BGR to RGB
  and collected; after the mp4 is finalized, the frames are stacked into one
  `(n_frames, H, W, 3)` uint8 array and written to `<out_dir>/video.npy`. The
  mp4 itself is unaffected -- it still gets the original BGR array. Ground
  truth's `npy_path` key records the array's path, or `null` when
  `save_npy=False`.
- `scripts/smooth_pursuit/generate_samples.py` gained a `--save-npy` CLI flag
  wired straight through to `generate_stimulus`.
- Verified end to end: frame count in `video.npy` matches the mp4's decoded
  frame count (200 frames at the current config's fps/phase durations),
  array dtype/range is uint8 in [0, 235] as expected, and omitting the flag
  produces no `.npy` file with `npy_path: null` in `ground_truth.json`.

## 2. Array size checked after a real batch

The user generated a batch of samples with `--save-npy` on
(`data/smooth_pursuit_samples/`). Each `video.npy` is **~84 MB**
(200 frames x 384x384x3 uint8, uncompressed) versus the corresponding
`video.mp4` at **~21 KB** -- roughly 4000x larger, since `np.save` stores raw
pixels with no compression at all. Flagged that this adds up fast at batch
scale (10 trials already ~840 MB) and that `np.savez_compressed` would trade
disk for a compression step -- which would undercut the point of this
artifact-isolation test, so raw `np.save` stays the right choice, just worth
using `--save-npy` deliberately rather than on every routine batch.

## Not yet done

- No actual mp4-vs-npy pixel comparison has been run yet -- this session only
  built and verified the *capability* to save both; the comparison itself
  (e.g. re-decoding `video.mp4` frame-by-frame and diffing against
  `video.npy`) is the next step.
- Local vLLM hosting of `qwen3.8-27b` (the motivating reason for wanting raw
  frames/fps control) hasn't been set up in this repo yet.
- pylyshyn arm's stimulus_gen.py has no equivalent `save_npy` support (out of
  scope today, per the user).
