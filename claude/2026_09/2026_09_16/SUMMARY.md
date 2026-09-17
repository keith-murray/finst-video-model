# Hard-stimulus pylyshyn sweeps across models, reasoning on/off, a files reorg, and an even harder rerun

## Part 1: all models on the heuristic-neutralized hard stimuli

Per `TODO.md`: last session found that pylyshyn trials where the nearest-neighbor position
heuristic *fails* are especially hard for `gemma-4-31b-it`, both locally and on OpenRouter
(all landed in a tight 58-67% band once the heuristic was neutralized to chance, vs. 90%+ on
heuristic-easy trials). Today asked whether this generalizes across models: a 100-trial
(50/50), n_objects=3, n_cued=1, heuristic-neutralized-to-chance set, stretched video,
reasoning off, tested across several OpenRouter models, with a cost estimate up front.

Model set: the 6 models from `speed_sweep_models` minus `google/gemma-4-31b-it:free` (shares
a noisy rate-limited pool per [[reference-openrouter-provider-routing-gotchas]]) -- confirmed
with the user via AskUserQuestion. Video variant: stretched-only, also per the user's choice.

**Pipeline** (new `scripts/pylyshyn/hard_stimuli/`, now reorganized -- see Part 3):
`generate_hard_stimuli.py` (two-phase heuristic-chance generator, adapted from the 09-14
n=4 version to n_objects=3, fast redirect/speed, writes both `video.mp4` and
`video_stretched.mp4`), `estimate_hard_sweep_cost.py` (live OpenRouter pricing + real pilot
calls appended into the real `results.csv` so no pilot spend is wasted),
`run_hard_stimuli_sweep.py` (the 5-model sweep, resumable), `aggregate_hard_stimuli_sweep.py`,
`plot_hard_stimuli_sweep.py`.

**Result**: 100/100 trials scored, heuristic reverified at exactly 50.0%.

| Model | Accuracy | d' | n | Cost |
|---|---|---|---|---|
| qwen3.6-plus | 76.0% | 1.40 | 100 | $0.26 |
| gemma-4-31b-it | 73.0% | 1.27 | 100 | $0.07 |
| qwen3.5-397b-a17b | 70.4% | 0.96 | 54 scored (46 unparseable) | $0.16 |
| kimi-k3 | 60.6% | 0.53 | 99 | $1.55 |
| glm-5v-turbo | 54.0% | 0.20 | 100 | $0.45 |

Total actual cost $2.49 (estimate had been $1.5-2.8). Every model landed well below the
80-100%+ ceilings seen on heuristic-easy pylyshyn sets, generalizing 09-14's single-model
finding: once the position shortcut is neutralized, none of these models show the strong
"tracking" performance the easy stimuli suggested.

**Data-quality notes**: qwen3.5-397b-a17b produced garbled/truncated non-answers (e.g. "the
answer the answer") on 46/100 trials -- correctly caught as unparseable, not a scoring bug.
kimi-k3 lost one trial to a persistent 504 timeout across 3 attempts, accepted at n=99. Also
fixed a real routing bug: `google/gemma-4-31b-it` was occasionally falling back to a
`together` endpoint that 400s (needs a dedicated deployment) -- added a provider exclusion.

## Part 2: turning reasoning on ("medium" effort, uniformly)

TODO: "none of the models get above 80%... with thinking off... let's turn it on... medium
thinking setting across all models... I leave it to you to determine a fair way to balance
thinking token budgets." Fairness approach: the same literal `reasoning={"effort": "medium"}`
dict for every model (live-confirmed via `/models/{author}/{slug}/endpoints` that all 5 list
`reasoning` as a supported, non-mandatory parameter) -- matches this project's existing
reasoning-effort-sweep convention, since there's no cross-vendor token-budget equivalence to
match instead.

**A real detour**: the user asked for a cost estimate first (rightly suspecting it'd be
higher). Actual $ cost turned out modest (~$4.2 estimated, $6.55 actual) -- the real
bottleneck was **latency**, not price: a single gemma call via DeepInfra took 298.8s and
495.7s on two separate tries (only 497-1123 reasoning tokens -- genuinely slow serving, not a
large reasoning budget). Live-tested all 5 of gemma's candidate hosts at reasoning=medium:
`modelrun` was both fast (4.5s) and trustworthy (normal ~2531 prompt_tokens), `coreweave` was
fast but returned prompt_tokens=216 (a known token-accounting unreliability), `crusoe` 429'd,
`parasail` was clean but slower (67.9s). Repinned gemma to `modelrun` only and launched the
full sweep (concurrency=16, ~35 min wall-clock for all 500 calls end to end).

New `run_hard_stimuli_thinking_sweep.py` (reasoning=medium, MAX_TOKENS=16384, same 100-trial
stimulus set and stretched-video choice as Part 1), `estimate_hard_sweep_thinking_cost.py`,
`aggregate_hard_stimuli_thinking_sweep.py`, `plot_hard_stimuli_thinking_sweep.py` (grouped
bar chart, Part 1 vs. Part 2 side by side).

**Result**: reasoning closed the gap for 4 of 5 models.

| Model | No reasoning | Medium reasoning | Delta | Reasoning cost |
|---|---|---|---|---|
| qwen3.6-plus | 76.0% (d'=1.40) | **93.0%** (d'=2.84) | +17.0pp | $0.78 |
| qwen3.5-397b-a17b | 70.4% (d'=0.96) | **90.7%** (d'=2.59) | +20.4pp | $1.42 |
| kimi-k3 | 60.6% (d'=0.53) | **89.0%** (d'=2.45) | +28.4pp | $3.08 |
| gemma-4-31b-it | 73.0% (d'=1.27) | **88.0%** (d'=2.28) | +15.0pp | $0.31 |
| glm-5v-turbo | 54.0% (d'=0.20) | 64.0% (d'=0.73) | +10.0pp | $0.96 |

Total Part 2 cost $6.55. **glm-5v-turbo is the outlier** -- reasoning barely helped it (still
well short of 80%, d'=0.73 vs. the others' 2.3-2.8), suggesting it isn't using its reasoning
budget effectively on this task the way the other four are -- not yet investigated further.
qwen3.5-397b-a17b landed at n=98/100: two specific trials deterministically burned the entire
16384-token budget on reasoning alone (`finish_reason='length'`) across 3 identical attempts,
accepted as a real model/config limitation rather than retried further.

## Part 3: reorganizing scripts/pylyshyn and results/pylyshyn

Per TODO's "we will most likely pause this project indefinitely... let's organize the files."
Both directories (36 scripts, 17 result dirs) regrouped into the same four-category taxonomy:

- `scaling_sweeps/` -- speed/redirect x n_objects scaling (`speed_sweep`, `speed_sweep_models`,
  `nobjects_sweep{,_n4,_n5}`)
- `reasoning_sweeps/` -- qwen3.8 reasoning-effort work (`qwen3.8-27b{,-low}`, `qwen3.8-max`,
  `reasoning_sweep`, `stretch_sweep{,_prefast_archive}`)
- `gemma_local/` -- local-cluster-vs-OpenRouter and host comparisons (`gemma_local_*`,
  `host_sweep`)
- `hard_stimuli/` -- today's Part 1/2 work
- `archive/test/` -- one old smoke-test run, moved out of the way per the user's choice (not
  deleted)

`scripts/pylyshyn/` root now holds only the 3 modules imported everywhere
(`run_pylyshyn_trial.py`, `nearest_neighbor_heuristic.py`, `generate_samples.py`); every other
script moved into its category. `data/pylyshyn/` was left untouched (not in scope).

**Verified, not just moved**: every `RESULTS_DIR`/`TRIALS_ROOT`/`OUT_PATH`/dynamic
`results_dir = os.path.join(...)` constant updated to the new nested `results/pylyshyn/`
paths (data paths unchanged); added a one-line `sys.path` bootstrap to the 21 scripts that
import a now-cross-directory shared module; fixed every stale `Usage:` docstring line and
cross-file docstring reference. Ran a full import smoke test across all 36 scripts plus
`--help`/real invocations on a sample -- clean except the 2 scripts needing `transformers`
(cluster-only, pre-existing, unrelated to the reorg). Committed by the user.

## Part 4: an even harder stimulus set, new model set, both conditions rerun

"After work thoughts" follow-up, appended to `TODO.md` after Parts 1-3 were done: rerun the
sweep (both reasoning off and reasoning="medium") on a stimulus set where the nearest-neighbor
heuristic is **wrong on every single trial** (not just neutralized to chance), swapping
`qwen/qwen3.5-397b-a17b` -> `google/gemini-3.8-flash` and `z-ai/glm-5v-turbo` ->
`qwen/qwen3.8-27b`. User asked for a cost estimate first, expecting $9-10.

**A real model conflict surfaced immediately**: live-checking `google/gemini-3.8-flash` via
OpenRouter's `/models` endpoint found `reasoning.mandatory=true` (supported efforts
high/medium/low, no way to disable) -- it cannot run in the no-reasoning condition at all.
Also checked the wider Gemini family for the user (asked "is there a gemini model where
reasoning is optional?"): `gemini-3-flash-preview`, `gemini-3.1-flash-lite{,-preview}`,
`gemini-2.5-flash{,-lite}` all have optional reasoning and video support -- every other 3.x
Gemini model (3.5/3.6/3.7/3.8-flash, 3.1-pro) is mandatory. Per the user's choice, kept
`gemini-3.8-flash` as named but dropped it from the no-reasoning pass entirely (4 models
there: gemma-4-31b-it, qwen3.6-plus, kimi-k3, qwen3.8-27b) rather than substituting a
different model or approximating "off" with low effort; it's added back for the
reasoning="medium" pass (5 models total). `qwen/qwen3.8-27b` confirmed non-mandatory
(`{"mandatory": false, "supported_efforts": ["xhigh","medium","low"]}`, matches
[[reference-local-qwen-reasoning-effort]]'s no-"high" finding for this model family).

**Pipeline**: `generate_hardest_stimuli.py` (2-bucket variant of Part 1's generator -- only
keeps seeds where `heuristic_predicts_match() != probe_is_target`, 50 per
probe_is_target class instead of 25, so the heuristic hits exactly 0.0% instead of 50.0%),
`run_hardest_stimuli_sweep.py` (no-reasoning, 4 models), `run_hardest_stimuli_thinking_sweep.py`
(reasoning=medium, 5 models, gemma re-pinned to `modelrun` per Part 2's fix),
`estimate_hardest_sweep_cost.py`/`estimate_hardest_sweep_thinking_cost.py`,
`aggregate_hardest_stimuli_sweep.py`/`_thinking_sweep.py`, `plot_hardest_stimuli_sweep.py`/
`_thinking_sweep.py` (the latter drawn as a union, not intersection, of both passes' models so
gemini-3.8-flash still appears as a reasoning-only bar rather than being dropped from the
figure).

**Cost estimate**: $2.10 (no-reasoning) + $5.86 (medium-reasoning) = **$7.96** total -- both
pilot passes came back completely clean (0 errors) on the first try, including both untested
models. Confirmed with the user before launching either full sweep.

**Result**: 100/100 trials scored per model, heuristic reverified at exactly 0.0%.

No-reasoning pass (400/400 clean, first try):

| Model | Accuracy | d' | Cost |
|---|---|---|---|
| kimi-k3 | 60.0% | 0.50 | $1.60 |
| gemma-4-31b-it | 56.0% | 0.35 | $0.02 |
| qwen3.6-plus | 51.0% | 0.05 | $0.26 |
| qwen3.8-27b | 47.0% | **-0.18** | $0.15 |

Every model landed at or below chance -- qwen3.8-27b actually scored *worse* than the
heuristic, an even more decisive null result than Part 1's chance-neutralized set (where
qwen3.6-plus/gemma still cleared 70%+).

Medium-reasoning pass (500/500 clean after one backfill for 8 transient
`prompt_tokens=216` video-attachment errors, same class as Part 2's):

| Model | No reasoning | Medium reasoning | Delta | Cost |
|---|---|---|---|---|
| **gemini-3.8-flash** | *(mandatory reasoning, N/A)* | **100.0%** (d'=4.67) | -- | $1.04 |
| qwen3.6-plus | 51.0% (d'=0.05) | 87.0% (d'=2.24) | +36.0pp | $0.92 |
| kimi-k3 | 60.0% (d'=0.50) | 84.0% (d'=1.99) | +24.0pp | $3.40 |
| qwen3.8-27b | 47.0% (d'=-0.18) | 76.8% (d'=1.43) | +29.8pp | $0.70 |
| gemma-4-31b-it | 56.0% (d'=0.35) | 77.0% (d'=1.54) | +21.0pp | $0.32 |

Total actual cost **$8.40** ($2.03 + $6.37) -- close to both the $7.96 pre-run estimate and
the user's $9-10 expectation. `gemini-3.8-flash`'s **perfect 100% accuracy (d'=4.67)** is the
standout finding of the whole session -- dramatically ahead of every other model on the
hardest stimulus set built so far, though the mandatory-reasoning constraint means it isn't a
clean apples-to-apples comparison the way the other four models' off/on pairs are.

**Comparing to Parts 1/2** for the three overlapping models (heuristic-neutralized-to-chance
-> heuristic-always-wrong): gemma 73.0%->56.0%, qwen3.6-plus 76.0%->51.0% (collapsed to
chance), kimi-k3 60.6%->60.0% (unchanged -- it was already near its floor). Under reasoning:
gemma 88.0%->77.0%, qwen3.6-plus 93.0%->87.0%, kimi-k3 89.0%->84.0% -- reasoning still
recovers most models substantially on the harder set, just a bit less than on the
chance-neutralized one.

## Open threads (carried over / updated)

- **`gemini-3.8-flash`'s perfect 100% (d'=4.67) on the hardest stimulus set** is the most
  striking unexplained result of the session -- worth a dedicated follow-up (e.g. does it hold
  up at lower reasoning effort, or on Part 1/2's chance-neutralized set too?) if the project
  resumes, keeping in mind it's confounded with reasoning being mandatory for this model.
- **glm-5v-turbo's weak response to reasoning** (Part 2) is still unresolved -- worth checking
  whether it's actually using its reasoning budget productively (token counts were comparable
  to the other models: ~1206 mean reasoning tokens) or just generating unproductive text. Note
  it was dropped from the Part 4 model set, so no further data on it this session.
- Project is paused indefinitely as of this session per the user -- `smooth_pursuit` arm
  remains untouched since 2026-08-14, and the local-vs-OpenRouter gemma gap
  (image-preprocessing/checkpoint-identity candidates from 09-09/09-10) was never resolved.
- The heuristic-always-wrong construction (Part 4) is a cleaner, more decisive version of
  Part 1's heuristic-neutralized-to-chance one (also now proven at n_objects=3/4) --
  probably the single most reusable methodological tool this project built, worth leading
  with if the project resumes.
