# Local gemma-4-31b-it pylyshyn: native mp4 input pipeline

## Task

Per `TODO.md`: the user got `torchcodec`+`ffmpeg` working on the department cluster
(`claude/skills/cluster/ffmpeg/test_torchcodec_with_module.sh`, validated by decoding an
existing trial's `video.mp4`). Built a sweep to test whether feeding the locally-hosted
gemma-4-31b-it its `video.mp4` directly -- instead of pre-decoded `video.npy` frames plus a
hand-built `video_metadata` dict -- narrows the accuracy gap vs. OpenRouter's hosted version
seen in `2026_09_10`'s pylyshyn run.

## Pipeline built

Sibling scripts to the existing npy-based ones, reusing the same 60 trials
(`data/pylyshyn/gemma_local_nobjects_sweep/trials/`, n_objects={3,4,5} x seed 0-19, fast
redirect/fast speed, native only) -- no new stimulus generation needed since `video.mp4`
already exists in every trial dir.

- `scripts/pylyshyn/run_gemma_local_batch_mp4.py` -- feeds `{"type": "video", "path":
  video_path}` directly into `processor.apply_chat_template(..., num_frames=32)`, no
  `video_metadata`/`do_sample_frames`, per HF's chat-templating-multimodal docs (path-based
  video loading is decoded via torchcodec automatically). Writes `cluster_response_mp4.json`
  (distinct filename from the npy script's `cluster_response.json`, same trial dirs, so
  neither run clobbers the other).
- `slurm/run_gemma_pylyshyn_local_mp4.sh` -- same job shape as the npy variant, adds
  `LD_LIBRARY_PATH="$HOME/micromamba/envs/ffmpeg-env/lib:..."` before activating the venv
  (from the user's torchcodec test script) so mp4 decoding works.
- `scripts/pylyshyn/aggregate_gemma_local_mp4_nobjects_sweep.py` -- scores via
  `finst_video_model.scoring`, writes `results/pylyshyn/gemma_local_mp4_nobjects_sweep/
  {results.csv,config.json}`.
- `scripts/pylyshyn/plot_gemma_local_mp4_vs_all.py` -- 4-series figure: OpenRouter,
  Local-mp4 (new), Local-npy (2026-09-10's run), nearest-neighbor heuristic.

**Workflow note**: code moved via `git commit`/`push` locally + `git pull` on the cluster-side
clone, not rsync (rsync is data-only in this project) -- corrected mid-session after drafting
rsync-based push instructions in the new scripts' docstrings; fixed those docstrings after the
fact. Data (trial dirs incl. `video.mp4`) was rsynced to the cluster ahead of this session by
the user. As with 2026-09-09/09-10, no cluster SSH access this session -- the user ran
push/sbatch/pull/aggregate/plot end-to-end themselves and delivered the finished
`results/pylyshyn/gemma_local_mp4_nobjects_sweep/` directly.

## Result: mp4 helps at n_objects=4 only, not uniformly

60/60 trials scored, 0 errors, 0 unparseable.

| n_objects | OpenRouter | Local, mp4 (new) | Local, npy (old) | Heuristic |
|---|---|---|---|---|
| 3 | 90% | 75% | 75% | 85% |
| 4 | 80% | **95%** | 65% | 75% |
| 5 | 90% | 65% | 65% | 75% |

At n_objects=4, switching to native-mp4 input produced a large, clean jump (65%->95%),
overtaking both OpenRouter and the heuristic. At n=3 and n=5, mp4 landed on the *exact same*
accuracy as the old npy run and stayed well below OpenRouter -- the video-input pipeline
change did not generalize.

Per-trial agreement (joined on matching `n_objects`+`seed`, since the npy run's trial dirs
were a separate generation with different random `trial_id`s than what's on disk today --
same deterministic stimulus content, different UUIDs, a known-harmless quirk per
[[project-status-2026-09-10]]): mp4 and npy gave the same True/False answer on 44/60 trials --
a real, concentrated divergence rather than uniform noise.

**Read on the open local-vs-OpenRouter gap**: this weakens "video preprocessing pipeline
mismatch" as a complete standalone explanation -- if that were the whole story, native mp4
input should have closed the gap broadly, not just at one n_objects value. The
checkpoint-identity candidate from [[project-status-2026-09-09]]/[[project-status-2026-09-10]]
remains the more likely explanation for the residual gap at n=3/n=5; mp4-vs-npy input
encoding does appear to be a real, if partial, contributor (n=4's jump is too large and clean
to be noise).

## Task 2: does the n=4 mp4 jump hold at higher power with the heuristic neutralized?

The 65%->95% jump at n=4 was a single striking point on a 20-trial (10/10) sample, and the
nearest-neighbor heuristic already scored above chance (75%) on that same sample -- both
reasons to distrust it before leaning on it. Built a follow-up per `TODO.md`'s Task 2: a
100-trial (50/50) n_objects=4 set deliberately constructed so the heuristic sits at exactly
chance, then reran both pipelines on it.

- `scripts/pylyshyn/generate_gemma_local_n4_heuristic_chance_stimuli.py` -- two-phase
  generator: Phase 1 scans seeds (parity-based 50/50 target/distractor split), scoring each
  candidate against `nearest_neighbor_heuristic.heuristic_predicts_match` inside a throwaway
  `tempfile.TemporaryDirectory()` (no `video.npy` written for rejects, avoiding ~44MB/trial of
  disk waste), and buckets seeds into `(probe_is_target, heuristic_correct)` groups; stops
  once each of the 4 buckets has 25 seeds (100 total, heuristic accuracy on the union = exactly
  50/100 by construction). Phase 2 regenerates the 100 winning seeds for real (`video.mp4` +
  `video.npy` + `ground_truth.json`), asserting determinism against Phase 1's recorded result
  per seed, then independently re-verifies the whole set's heuristic accuracy via the actual
  `heuristic_accuracy_by_group` production function (not just its own bucketing arithmetic).
  Smoke-tested at small scale (3/bucket) before the full run; full run took ~73s locally,
  produced exactly 25/25/25/25, heuristic verified at 50.0%.
- Reused `run_gemma_local_batch.py`/`run_gemma_local_batch_mp4.py` and both existing aggregate
  scripts completely unchanged (all already took `--trials-root`/`--out-root`) -- only new
  code needed was the generator, two new slurm scripts
  (`slurm/run_gemma_pylyshyn_local_n4_chance{,_mp4}.sh`), and a new bar-plot script
  (`scripts/pylyshyn/plot_gemma_local_n4_heuristic_chance_bar.py`).
- Bar-plot error bars: switched from population Bernoulli std (`sqrt(p*(1-p))*100`, the
  initial request) to SEM (`sqrt(p*(1-p)/n)*100`, matching every other pylyshyn plot in this
  repo) per explicit follow-up request after the first version was already generated.

### Result: the dramatic jump does not replicate

| | Accuracy | SEM | n |
|---|---|---|---|
| Local, npy | 58.0% | 4.9% | 100 |
| Local, mp4 | 65.0% | 4.8% | 100 |
| Heuristic | 50.0% | 5.0% | 100 |

At higher power with the heuristic neutralized, both pipelines land modestly above chance and
above the heuristic, and mp4 still edges out npy -- but the gap shrank from +30pp (65%->95% on
20 trials) to +7pp (58%->65% on 100 trials), well within one SEM of each other, not a
statistically distinguishable difference at this n. The original 95% reading was very likely
an artifact of a small sample combined with the heuristic's above-chance edge on that
particular 20-trial set, not a robust effect of switching to native mp4 input.
`results/pylyshyn/gemma_local_n4_heuristic_chance/accuracy_bar_npy_vs_mp4_vs_heuristic.png`.

## Task 3: the same 100 trials, run on OpenRouter -- for completion

Per `TODO.md`'s Task 3 ("just for completion"): ran the identical 100 heuristic-neutralized
trials from Task 2 through OpenRouter's hosted `google/gemma-4-31b-it`, one host/one condition
(no sweep), added as a fourth bar to Task 2's plot.

- `scripts/pylyshyn/run_gemma_local_n4_heuristic_chance_openrouter.py` -- reuses the exact
  `video.mp4` files already on disk (no regeneration), `run_pylyshyn_trial.build_question` for
  byte-identical question wording, same `MODEL`/`REASONING`/`MAX_TOKENS` convention as
  `run_pylyshyn_host_sweep.py`.
- **Hit a real bug**: pinned to deepinfra-fp8 first (top scorer in the 09-10 host sweep) --
  100/100 trials failed identically. Root cause: OpenRouter returns 200 OK with an embedded
  `error` object (`"Upstream error from DeepInfra: ...Rate limit exceeded..."`) when a pinned
  provider (`allow_fallbacks: False`) is rate-limited on its own end -- `ask_about_video`'s
  `response.raise_for_status()` never saw this since the HTTP status is 200, so it crashed on
  `data["choices"]` with an opaque `KeyError: 'choices'`. Fixed in
  `src/finst_video_model/vlm_client.py`: detects a missing `"choices"` key, retries (same
  `MAX_429_RETRIES` budget, new `PROVIDER_ERROR_BACKOFF_S=20.0`) if the embedded error looks
  rate-limit/provider-unavailable-shaped, else raises a clear `RuntimeError`. deepinfra-fp8
  stayed down even after retries (a persistent account-level limit, not a transient burst --
  same class of issue as 09-10's Crusoe 429s); live-tested alternatives and switched to
  **deepinfra-fp4** (verified healthy), which completed cleanly.
- Result: 100/100 scored, 0 errors, 0 unparseable, **67.0%** accuracy (SEM 4.7%), d'=1.04, all
  confirmed `served_by_provider=DeepInfra`.
- `scripts/pylyshyn/plot_gemma_local_n4_heuristic_chance_bar.py` extended to 4 bars (npy/mp4/
  OpenRouter/heuristic), widened figure + two-line labels to fit.

### Updated picture

| | Accuracy | SEM | n |
|---|---|---|---|
| Local, npy | 58.0% | 4.9% | 100 |
| Local, mp4 | 65.0% | 4.8% | 100 |
| OpenRouter (deepinfra-fp4) | 67.0% | 4.7% | 100 |
| Heuristic | 50.0% | 5.0% | 100 |

All three model conditions land in a tight band (58-67%), well within 1 SEM of each other --
OpenRouter is nominally highest but not distinguishably so from local-mp4, and local-npy is
not distinguishably lower either. This is a much smaller, noisier gap than the 20-trial n=4
sample suggested at any point today (OpenRouter 80%, local-mp4 95%, local-npy 65% there) --
consistent with Task 2's finding that the small-sample readings were not reliable.
`results/pylyshyn/gemma_local_n4_heuristic_chance/accuracy_bar_npy_vs_mp4_vs_heuristic.png`.

### Follow-up: the "poor" OpenRouter numbers track the heuristic split, not a generic difficulty bump

The user's own observation, checked post-hoc: since the 100-trial set is exactly 50
heuristic-correct/50 heuristic-incorrect by construction (Task 2), split each model's
performance by that same label (heuristic correctness recomputed per trial via
`heuristic_predicts_match`, joined against each results.csv by `trial_id`):

| | Accuracy, heuristic-correct half | Accuracy, heuristic-incorrect half | d', heuristic-correct | d', heuristic-incorrect |
|---|---|---|---|---|
| Local, npy | 64.0% | 52.0% | 0.75 | 0.10 |
| Local, mp4 | 72.0% | 58.0% | 1.15 | 0.40 |
| OpenRouter | 86.0% | 48.0% | **2.63** | **-0.13** |

OpenRouter's gemma is genuinely strong (d'=2.63) on trials where the nearest-neighbor
heuristic also lands correctly, but sits at chance-or-below (d'=-0.13) on the half where the
heuristic is wrong -- a 38pp accuracy swing and the starkest split of the three. Both local
pipelines show the same direction, just less extreme. This isn't just "we made the task
harder in some generic sense" -- performance specifically tracks whether a trial is one the
positional heuristic can solve, suggesting OpenRouter's gemma (and to a lesser extent the
local pipelines) may be leaning heavily on the same spatial-proximity cue as the naive
heuristic rather than doing genuine identity-preserving tracking. Directly relevant to
[[user-mech-interp-goal]]'s end goal (mech interp of tracking capacity, not benchmarking) --
this is evidence that at least part of what looks like "tracking ability" in these models'
raw accuracy numbers is actually a position-matching shortcut, echoing
[[project-status-2026-09-01]]'s original motivation for building the heuristic-comparison arm
in the first place.

## Open threads (carried over / updated)

- Local-vs-OpenRouter gap: at n_objects=4 with the heuristic neutralized and n=100, local-npy/
  local-mp4/OpenRouter are NOT distinguishably different (58/65/67%, all within 1 SEM) --
  today's entire "mp4 input closes the gap" thread downgrades to "not established, likely a
  small-sample artifact of the original 20-trial readings across the board, OpenRouter
  included." Checkpoint-identity remains the leading untested candidate for the gap seen at
  n=3/n=5 in Task 1's 60-trial run, which this session didn't revisit.
- New thread: all three model conditions' accuracy is much more strongly explained by whether
  a trial is heuristic-solvable (d' 0.75-2.63) vs. not (d' -0.13-0.40) than by which
  input pipeline/host served it -- OpenRouter's own tracking-like accuracy may be
  substantially a spatial-proximity shortcut rather than genuine identity tracking. Worth a
  dedicated follow-up (e.g. a full heuristic-correct-vs-incorrect split across n_objects/
  models, not just this one n=4 sample) before leaning on any raw accuracy number in this
  arm going forward.
- `vlm_client.ask_about_video` now handles 200-OK-with-embedded-error responses from a pinned
  provider (retries, then raises clearly) -- worth remembering for any future single-provider
  pin, not just this task.
- `smooth_pursuit` arm still untouched since 2026-08-14.
