# Summary: local gemma-4-31b-it on pylyshyn, a precision-mismatch finding, and a 7-host OpenRouter comparison

Full context/goals: `claude/2026_09/2026_09_10/TODO.md`. Two tasks, the
second added as a same-session continuation into the next calendar day.

## Task 1: local cluster gemma-4-31b-it on pylyshyn

Built and ran the full pipeline (this Mac still has no SSH access to
`<cluster-host>` -- confirmed again live via a failed `rsync`/`ssh` attempt, same
as `claude/2026_09/2026_09_09/SUMMARY.md`; the user/on-cluster session ran
the actual push/sbatch/pull/aggregate/plot steps and delivered
`results/pylyshyn/gemma_local_nobjects_sweep/`):

- `scripts/pylyshyn/generate_gemma_local_stimuli.py` -- generates the
  60-trial set (n_objects={3,4,5} x seeds 0-19, fast redirect/fast speed,
  native only), writing `video.npy` alongside the usual `video.mp4`/
  `ground_truth.json` (pylyshyn's `generate_stimulus`, unlike
  debug_circular's, doesn't persist a `.npy` itself).
- `scripts/pylyshyn/run_gemma_local_batch.py` -- cluster-side, self-contained,
  mirrors the debug_circular gemma cluster scripts: 32 frames
  (`do_sample_frames=True, num_frames=32`, matching OpenRouter's confirmed
  fixed cap for this model), HF "recommended" sampling.
- `slurm/run_gemma_pylyshyn_local.sh`, `scripts/pylyshyn/
  aggregate_gemma_local_nobjects_sweep.py`, `scripts/pylyshyn/
  plot_gemma_local_vs_openrouter.py`.

**Result: the local-vs-OpenRouter gap replicates on a second, structurally
different stimulus, and local now scores *below* the nearest-neighbor
heuristic at every n_objects.**

| n_objects | OpenRouter acc (d') | Local acc (d') | Heuristic acc |
|---|---|---|---|
| 3 | 90% (2.19) | 75% (1.22) | 85% |
| 4 | 80% (1.92) | 65% (0.70) | 75% |
| 5 | 90% (2.19) | 65% (0.87) | 75% |

60/60 trials scored, 0 errors, 0 unparseable. Checked the `predicted`
column for a 09-09-style total response-bias collapse: n_objects=3/4 are
close to balanced against the 10/10 ground-truth split; n_objects=5 shows a
real but modest conservative shift (15 False/5 True, hits=4/misses=6/
false_alarms=1/correct_rejections=9) -- a genuine weak-signal reading
(d'=0.87), not a fixed-answer artifact.

## Precision check: local runs bf16, OpenRouter runs fp4/fp8

User asked what precision the local model runs at. `dtype="auto"` in every
local gemma script resolves to whatever the checkpoint declares --
`test_gemma4_baseline.py`'s "~62GB model" comment implies bf16 (62GB /
30.7B params ~= 2.0 bytes/param). Checked OpenRouter's live
`/models/google/gemma-4-31b-it/endpoints` API: every provider confirmed to
serve video for this model runs **fp4 or fp8**, not bf16 -- the opposite
direction from what would naively explain local's *lower* accuracy (local
is higher precision and still does worse). Saved as
`reference_gemma_precision_mismatch` memory. Also discovered OpenRouter's
`provider.quantizations` request field, letting DeepInfra's 3 internal
serving tags (turbo=fp4, fp8, ultra=fp8) be split into two precisely-pinned
hosts instead of one ambiguous one.

## Task 2: does OpenRouter's own host/provider choice matter?

Built `scripts/debug_circular/run_debug_circular_host_sweep.py` and
`scripts/pylyshyn/run_pylyshyn_host_sweep.py` (model fixed at
`google/gemma-4-31b-it`, `host` a new grid axis pinning OpenRouter's
`provider` field, native only, every row logging `served_by_provider` to
catch a silent fallback), plus matching `plot_host_sweep.py` scripts in
each arm. Smoke-tested all 7 provider pins with one call each before the
real run -- all landed on their intended host, `quantizations` filter
accepted with no error.

**7 hosts tested** (debug_circular: rotation_deg 40/80/120, n_objects=3,
20 seeds/condition, 420 trials; pylyshyn: n_objects 3/4/5, fast redirect/
fast speed, 20 seeds/condition, 420 trials):

| host | quantization | context_length | prompt / completion price ($/M tok) |
|---|---|---|---|
| deepinfra-fp4 | fp4 | 262,144 | 0.09 / 0.34 |
| deepinfra-fp8 | fp8 | 262,144 (fp8 tag) / 131,072 (ultra tag) | 0.13-0.27 / 0.38-0.76 |
| coreweave | fp4 | 262,144 | 0.10 / 0.34 |
| crusoe | unknown (undisclosed) | 262,144 | 0.14 / 0.40 |
| parasail | fp8 | 262,144 | 0.15 / 0.40 |
| together | unknown (undisclosed) | 262,144 | 0.39 / 0.97 |
| modelrun | fp4 | 262,144 | 0.75 / 1.00 |

(Prices/quantization snapshotted live 2026-09-10; DeepInfra's fp8 pin can
land on either its "fp8" or "ultra" sub-tag, both reporting `quantization:
fp8` but different context/pricing -- not further disambiguated.)

**Result: hosts largely agree with each other, regardless of
quantization.** debug_circular's V-shaped rotation curve (near-ceiling at
40 deg, declining through 80, below chance by 120) reproduces on every
host with no clean split by fp4/fp8/unknown -- e.g. at rotation_deg=40,
d' ranges only 1.84-2.44 across all 7 hosts (deepinfra-fp4 2.16 vs.
deepinfra-fp8 2.19, its own two quantization tiers, are nearly identical).
pylyshyn shows the same pattern: accuracy stays in the 75-100% band for
every host at every n_objects, well above chance, with no visible
fp4-vs-fp8-vs-unknown clustering (`results/pylyshyn/host_sweep/
accuracy_by_nobjects_hosts.png`, `results/debug_circular/host_sweep/
accuracy_by_angle_hosts.png`).

**This is informative for the still-open local-vs-OpenRouter gap
investigation**: if OpenRouter's own hosts, spanning 3 different
quantization levels and clearly different serving stacks, all land in a
similar accuracy band, that band is likely a property of *how OpenRouter
serves this model* broadly (optimized inference stack, its own video
ingestion path) rather than being explained by precision specifically --
strengthens [[reference-gemma-precision-mismatch]]'s reading that
precision alone doesn't explain the local pipeline's much lower scores,
and keeps checkpoint identity / preprocessing pipeline as the more likely
open candidates.

## Data-quality note: Crusoe's real rate limit

Crusoe returned persistent `429 Too Many Requests` on a fixed set of 13
(n_objects, seed) pylyshyn combos across three separate attempts
(concurrent, low-concurrency, and fully serial retries, each already
using `ask_about_video`'s built-in `Retry-After`-based backoff) --
identical failures every time, not the usual random-subset flakiness seen
elsewhere in this project. Read as a real, longer-window account-level
quota rather than ordinary contention; continuing to hammer it seemed
unproductive. Accepted with reduced sample size for those two cells:
crusoe n_objects=4 at n=15/20, n_objects=5 at n=12/20 (both still
comfortably usable, just smaller than the other hosts' clean n=20).
Separately, an unrelated DNS-resolution outage (`Failed to resolve
'openrouter.ai'`) hit 103 trials mid-run early on -- a local network drop,
not provider-side; cleaned out and resumed without incident.

## Open threads (carried over / updated)

- The local-vs-OpenRouter gap's root cause is still unidentified. Today's
  host sweep weakens "precision" as a standalone explanation (OpenRouter's
  own fp4/fp8/bf16 spread doesn't split accuracy) without replacing it --
  [[project-status-2026-09-09]]'s image/preprocessing-mismatch and
  checkpoint-identity candidates remain the most promising next steps,
  now with more evidence pointing at the serving pipeline (not the number
  format) as the differentiator.
- `smooth_pursuit` arm still untouched since 2026-08-14.
