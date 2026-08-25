# Session summary — 2026-07-21

Finished the 100-trial multi-N sweep (`batch_id ce675830`) that was left
partway through last session, added concurrency + rate limiting to the batch
runner along the way, and got the project's first real capacity curve.

## Batch runner: concurrency + rate limiting

`scripts/run_comprehension_batch.py` gained a `--concurrency` flag (default
5): trials now run via a thread pool instead of one at a time, since each
trial's OpenRouter call is latency-bound rather than CPU-bound. Stimulus
generation and the VLM client already used only local, per-call RNG/file
state, so no locking was needed beyond keeping `results.csv` writes on the
main thread.

`src/finst_video_model/comprehension/vlm_client.py` now enforces
OpenRouter's 20 requests/minute limit itself, via a sliding-window rate
limiter shared across all threads in the process. This decouples
`--concurrency` from the actual request rate -- concurrency only controls
local parallelism now, since the real cap is centrally enforced regardless
of pool size.

This was prompted by the background batch job getting killed mid-run twice
(most likely the user's laptop sleeping/shutting down, not a harness
timeout -- see prior session's note). The remaining trials were finished
under `caffeinate -i` to prevent sleep, using the new concurrent path.

## Result: first real capacity curve

`results/ce675830/` (`qwen/qwen3.5-397b-a17b`, `n_circles=[2,3,4,6,8]`,
`n_cued=1`, seeds 0-19, 10s timing) -- exact-match identity accuracy vs.
chance (1/n_circles):

| n_circles | chance | accuracy |
| --------- | ------ | -------- |
| 2         | 50%    | 84%      |
| 3         | 33%    | 47%      |
| 4         | 25%    | 45%      |
| 6         | 17%    | 22%      |
| 8         | 13%    | 0%       |

This is a genuine declining capacity signal, not flat near-chance
performance: well above chance at n=2/3, collapsing to zero (below chance)
by n=8. This supersedes the earlier read from smaller single-N pilots
(~30% at n=4 vs. 25% chance, called statistically indistinguishable from
guessing) -- with the full multi-N sweep and 20 seeds/condition, the shape
across N is now clear, though n=3 vs. n=4 individually overlap heavily in
their SEM error bars (not resolved at that resolution). Plot at
`results/ce675830/accuracy_plot.png`.

5 trials (not 1, as first reported) errored with a JSON parse failure on
the OpenRouter response body rather than scoring -- 2 during the original
sequential run, 3 during the concurrent resume; intermittent and not
correlated with concurrency (95/100 scored cleanly).

## Retrospective test: "nearest-to-original-position" heuristic

Discussed an alternative hypothesis with the user (from a separate
conversation with Claude about the results): rather than really tracking
the cued circle, the model might just answer with whichever label ends up
nearest to the cued circle's *original* (cue-phase) position.

`scripts/analyze_position_heuristic.py` (new) tests this retrospectively on
the existing `ce675830` data, at zero extra API cost: it re-derives each
circle's deterministic start position from the trial's saved seed/config
(`build_circles` is pure, so no new video/API calls needed), finds which
label ends up nearest to the cued circle's start, and compares that
"heuristic" prediction against both the model's actual answer and the true
cued letter, split into agreement trials (heuristic == truth) vs. conflict
trials (heuristic != truth, the informative subset).

Result (95 scored trials, 31 agreement / 64 conflict): model accuracy is
65% on agreement trials but collapses to 28% on conflict trials, while
matching-the-heuristic rate on conflict trials is only marginally higher
(33%). Conclusion: strong evidence the model isn't doing robust identity
tracking (accuracy is propped up by cases where a shortcut happens to agree
with truth), but this specific heuristic isn't confirmed as *the*
mechanism -- the 33% vs. 28% gap is within noise at n=64. A deliberately
engineered conflict-trial manipulation (forcing heuristic/truth disagreement
by construction, rather than relying on the ~67% natural conflict rate)
would give a cleaner test; held off on this for now per the user. Full
row-level output at `results/ce675830/position_heuristic_analysis.csv`.

## Video codec fix: mp4v -> avc1

The user spotted that a smoke-test video played back as solid green.
Investigated: the pixel data itself was correct (confirmed by decoding the
same file via `cv2.VideoCapture` and extracting frames -- circles rendered
correctly), so this was a codec/player issue, not corrupted data.
`stimulus_gen.py`'s `cv2.VideoWriter` used the `mp4v` fourcc (raw MPEG-4
Part 2), which QuickTime/macOS's default player is notoriously bad at
decoding (often renders as solid green) even though standards-compliant
decoders (e.g. ffmpeg, which cv2.VideoCapture itself uses) read it fine.
Switched to `avc1` (H.264), confirmed to still open/write correctly and
confirmed by the user to play back correctly. Doesn't call prior results
into question (a genuinely garbled input wouldn't produce the
above-chance-then-declining capacity signal already seen), but removes any
lingering doubt and lets stimulus videos be visually sanity-checked going
forward.

## Two new sweeps: fps and speed_px_s

The user's own hypotheses, discussed with Claude separately: (1) fps
shouldn't matter much, since these VLMs likely downsample video internally
to ~1-2fps regardless of source frame rate; (2) slower circle speed should
improve tracking performance. `run_comprehension_batch.py` gained `--fps`
and `--speed-px-s` as new list-sweepable grid axes (alongside `n_circles`,
matching the existing pattern; both default to `TrialConfig`'s single
default value, so passing a list for just one sweeps only that axis).
`plot_comprehension_results.py` gained `--x-axis` to plot against any swept
column, not just `n_circles`.

Used a cheaper model for these, `qwen/qwen3.5-flash-02-23` (cheapest
non-free paid video-capable model on OpenRouter, ~6x cheaper than
`qwen3.5-397b-a17b`, already validated on this task in the first-ever
pilot), at `n_circles=4, n_cued=1`, 2s cue / 6s tracking / 2s label timing
(matching `ce675830`'s shortened timing), 10 seeds/condition.

**FPS sweep** (`results/52aac03d/`, fps=[1,2,4,8,24], speed_px_s=140 fixed):
accuracy 20%, 30%, 20%, 0%, 30% -- noisy, no clear trend, heavily
overlapping error bars at n=10/condition. Confirms the user's hypothesis:
no meaningful fps effect.

**Speed sweep** (`results/44f1823e/`, speed_px_s=[35,70,140,280], fps=24
fixed): accuracy 100%, 80%, 30%, 10% -- a clean, strong, monotonic decline,
much cleaner than either the fps sweep or the original n_circles capacity
curve. At the slowest speed the model is essentially perfect; at the
fastest it's at/below the 25% chance level for n_circles=4. Confirms the
user's hypothesis directly and is the cleanest effect found so far in this
project.

Plots at `results/52aac03d/accuracy_plot_fps.png` and
`results/44f1823e/accuracy_plot_speed_px_s.png`.

### Batch runner hardening along the way

- 7 trials in the fps sweep (all at fps=1, right at the start) and 1 in the
  speed sweep hit real errors: 6x `429 Too Many Requests` and 2x the
  familiar JSON-parse issue. The 429s were a genuinely new failure mode --
  traced to the rate limiter being per-process: a couple of quick smoke-test
  calls (separate script invocations) had just gone out moments before this
  sweep launched, so the account's real rolling window was already partway
  used up even though this fresh process's own deque started empty.
  `vlm_client.ask_about_video` now retries on 429 (respecting `Retry-After`,
  up to 3 attempts) instead of failing the trial outright.
- Backfilled all 8 dropped trials by stripping their error rows from
  `results.csv` and re-running `--resume` (both sweeps now 100% clean, 0
  errors).
- Real billed cost for the full day's `qwen3.5-flash-02-23` activity (smoke
  tests + both 50/40-trial sweeps + backfill, ~94 calls): **$0.325 total**,
  ~$0.0035/trial -- close to the pre-run estimate (scaled from
  `qwen3.5-397b-a17b`'s previously-observed real billing by the two models'
  live token-price ratio, since a fresh `/api/v1/key` usage delta right
  after a single call didn't update within 15s -- billing appears to lag).

## n_circles capacity curve re-run at slow speed (speed=70, fps=4)

Direct follow-up to the "does a slow-enough speed rescue accuracy even at
high n_circles?" question above. Re-ran the full `ce675830`-style n_circles
sweep (`results/e8489fbd/`, n_circles=2-8 with no gaps, n_cued=1, 10
seeds/condition, same 2s/6s/2s timing) but at `speed_px_s=70` and `fps=4`
instead of the original 140/24, still on `qwen/qwen3.5-flash-02-23`.

| n_circles | chance | accuracy |
| --------- | ------ | -------- |
| 2         | 50%    | 90%      |
| 3         | 33%    | 90%      |
| 4         | 25%    | 90%      |
| 5         | 20%    | 60%      |
| 6         | 17%    | 70%      |
| 7         | 14%    | 50%      |
| 8         | 13%    | 44%      |

Still declining, but **well above chance across the entire range**, even at
n=8 (44% vs. 13% chance) -- unlike the original `ce675830` curve (speed=140,
fps=24), which collapsed to 0% (below chance) by n=8. Confirms that the
original capacity-limit result was partly a speed confound, not a hard
object-count ceiling: slowing the circles down substantially rescues
performance even at high n_circles. 69/70 trials scored (1 hit the familiar
intermittent "response ended prematurely" error). Plot at
`results/e8489fbd/accuracy_plot.png`.

Re-ran the position-heuristic analysis on this batch too
(`results/e8489fbd/position_heuristic_analysis.csv`, 69 scored trials, 26
agreement / 43 conflict) -- a much cleaner and *opposite-direction* result
from the original fast-speed data: agreement-trial accuracy is 100%, and on
conflict trials the model matches **truth 53%** of the time vs. matches the
**heuristic only 14%** (vs. 28%/33%, statistically indistinguishable, in the
original speed=140 data). At this slower speed the model is clearly doing
real identity tracking rather than falling back on the position heuristic --
consistent with a story where the model leans on the heuristic more when
tracking is hard (fast motion) but shows genuine tracking ability once the
task is easier.

## n_circles x speed_px_s grid (five speed lines on one plot)

Direct follow-up to "narrow the speed sweep between 70-140px/s" above.
Extended `e8489fbd` (rather than starting a new batch) by hand-editing its
`results/e8489fbd/config.json` to add four more evenly-spaced speeds between
70 and 140 (`70, 87.5, 105, 122.5, 140`) alongside the existing `n_circles`
2-8 grid and 10 seeds/condition, all still at `fps=4`, then `--resume
e8489fbd` -- this reused the 70 already-collected speed=70 trials instead of
re-paying for them, since `--resume` skips any `(model, n_circles, fps,
speed_px_s, seed)` combo already in `results.csv`. Total grid: 350 trials
(7 n_circles x 5 speeds x 10 seeds).

`plot_comprehension_results.py` gained `--series-by` (default `model`) so a
plot's colored lines can split on any column, not just model -- needed here
to put `n_circles` on the x-axis with one line per `speed_px_s`. Expanded
the categorical palette from 3 to 5 slots (dataviz skill's slots 1-5: blue,
orange, aqua, yellow, magenta) and re-validated via
`scripts/validate_palette.js` for both light and dark surfaces before using
it (adjacent-pairlist checks all pass; light mode has a contrast WARN on 3
of the 5, satisfied by the existing plain-ink legend text rather than
colored labels).

Result (`results/e8489fbd/accuracy_plot_speed_px_s.png`, all 350 trials
scored, 0 errors after backfilling): the five speed curves **fan out as
n_circles increases** -- they cluster fairly close at n=2-4 (70-90% for the
three slowest speeds) but separate cleanly from n=5 onward, with slower
speed consistently outperforming faster speed at every n_circles level
(speed=70 on top throughout the range, speed=140 at the bottom). Confirms
the speed effect holds across the whole capacity curve, and that the
"rescue" from slowing down is strongest exactly where the task is hardest
(high n_circles) -- speed and object count jointly determine tracking
difficulty rather than acting as independent, additive factors.

### More batch-runner hardening: lid-close sleep, not just shutdown

This run got killed 4 times before completing (`--resume` each time). The
first kill showed a burst of network-level errors (`HTTPSConnectionPool`,
`Connection broken/aborted/reset`) rather than a clean process death --
different from the earlier 429/JSON-parse failure modes. Diagnosis:
`caffeinate -i` only blocks *idle* sleep, not the sleep triggered by closing
a MacBook's lid (clamshell sleep happens regardless of that assertion
unless the machine is in true clamshell mode with an external display). The
remaining 3 kills showed 0-1 errors each (the familiar intermittent
JSON-parse issue, unrelated), suggesting the lid stayed open after the first
interruption. Each kill was recovered the same way: strip error rows from
`results.csv`, `--resume` again. Total across all resumes: 350/350 trials,
only 1 real error along the way (the pre-existing intermittent one).

## Next steps

- Replicate the n_circles capacity curve on a second model (e.g.
  `google/gemini-2.5-flash`, already supported via `--models`) to see
  whether the capacity limit is qwen-specific or general.
- More seeds to resolve whether n=3 and n=4 are really different or just
  noise at this sample size.
- `n_cued` and `force_path_crossing` remain unswept axes if the
  single-cued-circle result needs deconfounding further.
- The deliberately-engineered conflict-trial manipulation (forcing
  heuristic/truth disagreement by construction) remains a natural follow-up
  to the position-heuristic analysis, if a cleaner causal test is wanted --
  now with an added angle: does the heuristic's explanatory power scale
  smoothly with speed, or is there a sharper transition?
- The 70-140px/s grid didn't reveal an obvious single "knee" -- the effect
  looks more like a smooth, continuous speed x n_circles interaction than a
  sharp threshold. Worth deciding whether a sharper knee exists outside this
  range (e.g. below 70 or above 140, crossed with n_circles) or whether the
  interaction really is smooth.
