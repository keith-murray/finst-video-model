# Session summary — 2026-07-17

Proof-of-concept goal from `TODO.md` is done: generate a stimulus image +
prompt, submit to Veo 3.1 through OpenRouter, and download the resulting
video. Automated evaluation is still out of scope (as planned).

## What got built

- **`src/finst_video_model/veo_client.py`** (new) — OpenRouter REST client
  for Veo 3.1, used instead of the Google Gemini SDK referenced in the
  original `TODO.md` (that reference came from an earlier, OpenRouter-blind
  conversation). Implements the submit -> poll -> download flow:
  - `submit_video_job` — POSTs prompt + frame-0 image (base64 data URI,
    `frame_images`/`first_frame`) to `POST /api/v1/videos`. Sets
    `generate_audio: False` to avoid the audio-safety false-positive
    failure mode called out in `TODO.md`.
  - `poll_job` — polls until a terminal status or timeout.
  - `download_video` — downloads the completed video (needs an
    `Authorization` header on the content URL — see Bugs below).
  - `run_trial` — chains stimulus generation, prompt building, submit,
    poll, and download for one `TrialConfig`, writing
    `frame0_<id>.png` / `ground_truth_<id>.json` / `video_<id>.mp4` /
    `generation_<id>.json` into a shared output directory, all linked by
    `trial_id`. Catches request/API failures into the result dict instead
    of raising, per the "log and skip" guidance in `TODO.md`.
- **`scripts/run_trial.py`** (new) — standalone script that runs one real
  trial via `run_trial`, writing output to `data/`.
- **Import fix** — `config.py`, `prompts.py`, `stimulus_gen.py` used
  bare `from config import ...` style imports that only worked when run as
  loose scripts from inside their own directory. Switched to
  `from finst_video_model.config import ...` so the package imports
  correctly once installed (editable install via `uv`).
- **Dependencies** — added `pillow`, `requests`, `python-dotenv` via `uv
  add`.
- **API key handling** — `OPENROUTER_API_KEY` is loaded from a local
  `.env` file (`python-dotenv`, loaded in `veo_client.py`). `.env` is
  gitignored; `.env.example` (empty key, committed) documents the expected
  variable name.
- **Output directory** — stimulus/video output goes to `data/`, which was
  already gitignored, so trial artifacts never hit git status.

## Bugs found and fixed

1. **Missing auth header on video download.** OpenRouter's own Python
   quickstart example fetches `unsigned_urls` without an `Authorization`
   header; in practice the content endpoint returns `401 Unauthorized`
   without one. Fixed by adding the header in `download_video`.
2. **Download failure was clobbering a successful generation status.** The
   original exception handling set `result["status"] = "error"` whenever
   *anything* in the try block failed, including a download error after
   Veo had already generated (and charged for) the video. This hid the
   fact that generation had actually succeeded. Fixed by wrapping the
   download step in its own try/except that writes to a separate
   `download_error` field, leaving `status` as whatever the generation
   job actually reported.

## Pricing correction

`TODO.md` and the OpenRouter docs snapshot in `claude/openrouter/` both
only had an illustrative example price ($0.50/video-second). Checked the
live `GET https://openrouter.ai/api/v1/videos/models` endpoint instead —
actual `google/veo-3.1` pricing (as of 2026-07-17):

| SKU                              | $/video-second |
| --------------------------------- | --------------- |
| without audio (720p/1080p)        | 0.20            |
| with audio (720p/1080p)           | 0.40            |
| without audio (4K)                | 0.40            |
| with audio (4K)                   | 0.60            |

At 8s/720p/no-audio, one trial costs **$1.60**.

## Real trial run

Ran one live trial end-to-end (`trial_id 7d7c2cbf`, `n_circles=6`,
`n_cued=2`, seed 42, 8s clip). Generation completed and was billed $1.60;
output is at `data/video_7d7c2cbf.mp4` with its linked
`ground_truth_7d7c2cbf.json` / `generation_7d7c2cbf.json`.

**Observation:** the output video showed noticeable hallucination relative
to the prompt (this was visually obvious on playback, not yet
quantified). Consistent with the design doc's caveat that Veo is
generative, not a physics simulator, and won't exactly honor the
prompt's motion/merge/split constraints. Whether this hallucination rate
is typical or unusually high for this trial is unknown — only one trial
has been run so far.

## Next steps (unchanged from `TODO.md`, still not started)

- Frame extraction from generated video (ffmpeg/opencv).
- Blob-tracking verification pipeline (detect, track, score identity
  accuracy) — will also need to account for the hallucination observed
  above (e.g. circle count drift) as part of the validity/exclusion
  checks already planned.
- Batch runner sweeping `n_circles`.
- Before scaling up: worth a small pilot batch (a handful of trials) to
  see whether the hallucination in this one trial was typical, and to get
  a real per-trial cost baseline for budgeting a full sweep (now known to
  be ~$1.60/trial at 8s/720p/no-audio, not ~$4).
