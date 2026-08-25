# Session summary — 2026-07-20

Big pivot day: went from "generate the whole clip with a video model" (Veo,
then video-extend) to "render our own ground-truth video and ask a
video-understanding VLM a text question" (the comprehension arm), wired that
arm all the way from stimulus to a real scored multi-model, multi-N sweep,
and hit (and fixed) a real batch-runner data-loss bug along the way.

## What got built

### Comprehension arm (`src/finst_video_model/comprehension/`)
Ported from the `claude_gen_files` draft (config.py, physics.py,
stimulus_gen.py), fixed to use absolute `finst_video_model.comprehension.x`
imports, and added `use_label_phase` as a hyperparameter (per `TODO.md`):
when `False`, the video ends after the tracking phase with no frozen/labeled
frame, for uses that don't need a text question (e.g. probing a frozen
encoder's activations directly). No `prompts.py` went into the package —
question/prompt text lives only in the calling scripts, per existing
convention (`README.md`'s note about `scripts/run_trial.py`).

- **`vlm_client.py`** — generic OpenRouter chat-completions wrapper
  (`ask_about_video`), sends the video as a base64 `video_url` data URL.
- **`scoring.py`** — `parse_answer_letters` (regex-extracts standalone
  capital letters from free text) and `score_answer` (exact-match,
  right-count, precision/recall against `ground_truth["cued_letters"]`).
- **`scripts/run_comprehension_trial.py`** — one full trial: stimulus ->
  question -> VLM call -> score, with `--seed`/`--model`/`--n-circles`/
  `--n-cued` CLI args for quick one-off comparisons.
- **`scripts/run_comprehension_batch.py`** — sweeps `n_circles` x `seeds` x
  `models` (now also `--cue-flash-s`/`--tracking-s`/`--label-s`). Reuses
  `run_comprehension_trial.py`'s `build_question` by importing it rather
  than duplicating.
- **`scripts/plot_comprehension_results.py`** — accuracy (exact-match) and
  mean recall vs. `n_circles`, one line per model with seed-to-seed error
  bars. Used the `dataviz` skill for the categorical palette (validated via
  `validate_palette.js`) and mark/legend conventions.

### Video-extend arm — built, then abandoned
`scripts/run_extend_trial.py` renders the unlabeled base video and an
`extend_prompt.txt` for a video-extension model to recolor the originally-
cued circles. Tried against Kling AI's video-extend feature: **it could not
actually extend the video** (per this file's own "Results" section above).
Combined with the Veo arm's earlier mixed results, the user decided to
**abandon video generation entirely** and focus solely on the comprehension
arm. Left in place as a record of the experiment; not developed further.

### Repo reorg
Siloed the original Veo generation-arm code (`client.py`, `config.py`,
`stimulus_gen.py`) into `src/finst_video_model/generation/`, mirroring
`comprehension/`, since both define a same-named `TrialConfig` with
incompatible fields. Updated `scripts/run_trial.py` / `run_circular_trial.py`
imports and README accordingly; marked that arm as abandoned in the README.

### Output layout: `data/` vs `results/`
Per the user's request, batch-runner output is now split by git-tracking
status: large/regeneratable per-trial artifacts (video, ground truth,
question, response) go to `data/<batch_id>/trials/<trial_id>/` (gitignored);
the small analysis output (`config.json`, `results.csv`, `accuracy_plot.png`)
goes to `results/<batch_id>/` (tracked in git).

## Bugs found and fixed

1. **`402` on any video-input OpenRouter call.** Not a pricing bug — turned
   out to be a flat "requires >= $1.00 balance for video" gate OpenRouter
   applies to any request with a `video_url`, independent of the actual
   (tiny, ~$0.001-0.03/call) per-request cost. Diagnosed by comparing a
   working text-only call against the same request with video attached via
   raw `curl`. Resolved once the user topped up.
2. **Batch runner data loss on interruption.** `run_comprehension_batch.py`
   only wrote its aggregate `results.csv` once, at the very end of the run.
   The 100-trial multi-N sweep got killed by what looks like a background-
   task duration limit (~56-60 min; a prior 43-min/20-trial run completed
   fine) after 38 real, paid trials had completed — stranding all 38 results
   in per-trial JSON files with no aggregate CSV. Recovered by reconstructing
   `results.csv` from each trial's `ground_truth.json` + `result.json`, then
   fixed the script for real: `results.csv` is now written incrementally
   (flushed after every trial), and the script gained `--resume <batch_id>`
   (skips already-completed model/n_circles/seed combos, reuses the
   original run's saved config) and `--limit N` (caps trials per invocation
   so future long sweeps can be safely chunked instead of risking another
   kill). Verified the fix live before resuming real spend.

## Pricing / cost notes

- OpenRouter's `google/gemini-2.5-flash` video-comprehension calls run
  ~$0.001-0.002/trial by token-rate math, but **observed actual billing**
  (checked via `/api/v1/key`'s `usage_daily`, before vs. after a batch of
  calls) came in higher, ~$0.015-0.03/trial depending on model — video
  tokenization in practice is denser than a naive tokens-per-second guess.
  Always calibrate against real billed usage before estimating a bigger run,
  not just token-rate arithmetic.

## Results (still very much a pilot, not a conclusion)

- **First 8-trial pilot** (`google/gemini-2.5-flash`, `qwen/qwen3.5-397b-a17b`,
  `qwen/qwen3.5-flash-02-23`; `n_circles=6, n_cued=2`, various seeds):
  1/8 exact match, mean recall ~0.31, every trial got the *count* right.
- **20-trial single-N sweep** (`results/99893293/`; `qwen/qwen3.5-397b-a17b`,
  `n_circles=4, n_cued=1`, seeds 0-19, default 11s timing): 30% exact match
  vs. 25% chance baseline — statistically indistinguishable from guessing,
  though `correct_count` was 100% (always answered exactly 1 letter, so the
  model understood the task format, it just isn't tracking identity).
  Discussed with the user: plausibly explained by video-VLMs sampling frames
  sparsely (e.g. ~1fps) rather than truly continuous motion, which would
  break identity tracking through a de-cued, featureless phase where no
  visual cue distinguishes the circles except their trajectory.
- **100-trial multi-N sweep** (`results/ce675830/`; same model,
  `n_circles=[2,3,4,6,8], n_cued=1`, seeds 0-19, shortened 10s timing
  (2s cue / 6s tracking / 2s label)): **58/100 complete** as of end of
  session (stopped deliberately partway through, to resume tomorrow via
  `--resume ce675830 --limit 20`). No aggregate accuracy-vs-N read yet —
  wait for all 100 before trusting the shape of the curve.

## Next steps

- Finish the multi-N sweep (`uv run python scripts/run_comprehension_batch.py
  --resume ce675830 --limit 20`, ~2-3 more chunks for the remaining 42
  trials), then run `plot_comprehension_results.py results/ce675830` for the
  actual accuracy-vs-`n_circles` capacity curve.
- Decide whether the near-chance result holds across N or whether smaller N
  (2-3 circles) shows any real tracking signal above chance — that
  distinguishes "no tracking ability at all" from "a real but very low
  capacity limit."
- Once a capacity curve exists for one model, worth comparing against
  Gemini 2.5 Flash and/or other models at the same grid, using the same
  `--models` sweep axis already supported.
- `n_cued` and `force_path_crossing` are still unswept axes (see
  `TrialConfig`) if the single-cued-circle result needs deconfounding
  further.
