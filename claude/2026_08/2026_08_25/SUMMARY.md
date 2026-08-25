# Summary: debug_circular stimulus + Qwen3.8-27B cluster pipeline

Full context/goals: `claude/2026_08/2026_08_25/TODO.md` -- with Qwen3.8-27B
now locally hosted on 2x A100s (full control over exactly what frames the
model sees, unlike OpenRouter's opaque server-side video preprocessing),
this session built and ran the simplest possible motion-perception stimulus
ahead of returning to the pylyshyn/smooth_pursuit arms, plus the full
cluster batch-inference pipeline to test it.

## 1. Repo flatten (task 1)

- Removed `src/finst_video_model/generation/` entirely (the abandoned Veo
  generation arm) and the 6 `scripts/prototype/*.py` files that depended on
  either it or the pre-pylyshyn/smooth_pursuit-split flat `comprehension`
  modules -- all already dead/broken before today.
- Flattened `src/finst_video_model/comprehension/*` up to
  `src/finst_video_model/*` directly (`pylyshyn/`, `smooth_pursuit/`,
  `scoring.py`, `vlm_client.py`), updating every import across `src/` and
  `scripts/`. Verified all core modules import cleanly and all remaining
  scripts run `--help` without error.

## 2. debug_circular stimulus (task 2)

- New `src/finst_video_model/debug_circular/` module: `n_objects` white
  crosses equally spaced around a circle, co-rotating as a rigid ring
  (no per-object heading, no minimum-separation logic needed -- equal
  spacing + rigid rotation makes collisions impossible by construction).
  Three phases matching the pylyshyn/smooth_pursuit probe-report
  convention: 1s cue (one cross red), 8s rotation (all white, direction/
  magnitude set by `clockwise`/`rotation_deg`), 1s probe (one cross red
  again, `probe_on_target` bool controls whether it's the same physical
  object cued at the start).
- Resolution was originally specified as 400x400 but changed to **384x384**
  by the user directly in `config.py` -- required to be an exact multiple
  of 32 so vLLM's vision encoder doesn't silently resize the video (would
  otherwise mis-size `max_model_len` and risk the same class of silent
  preprocessing bug the cluster handoff doc's Known Bug #5 already warns
  about for the `video_url` pathway).
- `scripts/debug_circular/generate_samples.py` sweeps `rotation_deg` x
  `probe_on_target` x `clockwise`, always writing `video.npy` +
  `ground_truth.json`, and `video.mp4` unless `--no-save-mp4` is passed --
  added specifically so stimuli can be generated directly on the cluster's
  compute nodes, which have no GPU video-encode device and can't reliably
  produce mp4s.
- 16 sample trials generated and eyeballed by the user at
  `data/debug_circular/debug_circular_samples/`.

## 3. Cluster batch-inference pipeline (task 3)

Built per an approved plan (`/Users/ktmurray/.claude/plans/tingly-swinging-thacker.md`),
grounded in `claude/skills/cluster/qwen38_cluster_handoff.md`'s
CONFIRMED-WORKING vLLM pattern:

- `scripts/debug_circular/run_cluster_batch.py` -- self-contained (no
  `finst_video_model` import; the cluster's `$HOME/qwen38/.venv` doesn't
  have this project's package), chunked + resumable (skips trials that
  already have a `cluster_response.json`, so a re-submitted job picks up
  where it left off rather than redoing work). Implements the TODO's
  mandated `chat_template_kwargs: {"enable_thinking": False}` return value,
  plus a belt-and-suspenders `enable_thinking=False` passed directly into
  `apply_chat_template(...)` at render time (since `chat_template_kwargs`
  alone was unconfirmed to have any effect once `prompt` is already a
  rendered string).
- `scripts/debug_circular/profile_cluster_timing.py` -- measures real
  **batched** per-trial timing (not serial, which wouldn't reflect vLLM's
  actual continuous-batching throughput) and checks whether thinking was
  actually suppressed.
- `slurm/profile_debug_circular.sh` and `slurm/run_debug_circular_batch.sh`
  -- sbatch wrappers mirroring `submit_video_job.sh`'s env setup.
- `scripts/debug_circular/aggregate_cluster_results.py` -- local-only
  script (imports `finst_video_model.scoring`) that turns rsynced-back raw
  `cluster_response.json` files into a scored, git-tracked
  `results/debug_circular/<run_name>/results.csv` + a d'-by-`rotation_deg`
  printout. Not yet run against real sweep data (see below).
- Also added `slurm/logs/` (tracked via `.gitkeep`) and a `slurm/logs/*.out`
  `.gitignore` rule so Slurm log files themselves don't get accidentally
  committed.

### Profiling job results (run today)

`slurm/profile_debug_circular.sh` ran successfully against the 16 existing
sample trials. Key numbers:
- Model load: ~400s (~6.7 min).
- Generation: ~1.3s/trial at chunk_size 10 or 16 (no real gain from bigger
  chunks at this scale).
- Extrapolated full 200-trial job: **~18 min total** (vs. the placeholder
  12-hour `--time` budget, which assumed the old reasoning-enabled
  ~200s/trial pacing -- thinking-disabled generation is drastically
  faster).
- `enable_thinking` suppression **confirmed working**: no `<think>` tag in
  any of the 16 outputs, and outputs were 2 tokens each.

**Finding worth tracking going forward**: all 16 profiled trials -- a mix
of rotation angles/directions/`probe_on_target` conditions that should
split ~8 True / 8 False by construction of the sweep -- came back with the
literal, identical response `'False'`. No variation at all. This implies
(assuming the expected ground-truth split) exactly 8 misses and 8 correct
rejections: **d' ≈ 0, chance-level, with a total "always say False"
response bias**, not partial signal. Prompt token counts were identical
and consistent (7808) across all 16, so this doesn't look like a pipeline
bug (video sizing/ingestion appears fine) -- more likely a genuine finding
consistent with this project's prior history of VLM response-bias effects
(the Gemini pilot's "opposite response biases masked by similar accuracy,"
which is exactly why this project scores with d' instead of raw accuracy).
Not confirmed either way yet -- the real 200-trial run's d'-by-`rotation_deg`
breakdown (once `aggregate_cluster_results.py` is run against it) will
either confirm this collapse or show it was an artifact of this small,
non-representative 16-trial sample.

### Real 200-trial sweep -- submitted, not yet resolved

The user generated real sweep data and submitted `run_debug_circular_batch.sh`
directly against the cluster's own copy of the repo, editing that
cluster-side copy's `--time` and `TRIALS_ROOT` by hand rather than through
this local git checkout. **The local repo's `slurm/run_debug_circular_batch.sh`
still has the original placeholder values (`--time=12:00:00`,
`TRIALS_ROOT=.../CHANGE_ME/trials`)** -- worth reconciling (pulling the
cluster-side edits back into this checked-in copy) once the job is checked
tomorrow, so the tracked version matches what was actually run.

## Not yet done

- Job outcome unknown as of this summary -- check tomorrow.
- No local copy of the real 200-trial `data/debug_circular/<run_name>/trials/`
  exists yet (only the original 16-trial `debug_circular_samples/`) -- it
  was generated directly on the cluster.
- `aggregate_cluster_results.py` has not been run against any real sweep
  output yet (only smoke-tested against 3 fabricated fake responses during
  development).
- The "always False" response-bias question above is open until real
  results come back.
- `smooth_pursuit`/`pylyshyn` arms untouched this session -- debug_circular
  was explicitly meant as a simpler sanity check before returning to them.
