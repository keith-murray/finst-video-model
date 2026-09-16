# Hard-stimulus pylyshyn sweep across models, reasoning on/off, and a files reorg

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

## Open threads (carried over / updated)

- **glm-5v-turbo's weak response to reasoning** is the most interesting unresolved thread --
  worth checking whether it's actually using its reasoning budget productively (token counts
  were comparable to the other models: ~1206 mean reasoning tokens) or just generating
  unproductive text.
- Project is paused indefinitely as of this session per the user -- `smooth_pursuit` arm
  remains untouched since 2026-08-14, and the local-vs-OpenRouter gemma gap
  (image-preprocessing/checkpoint-identity candidates from 09-09/09-10) was never resolved.
- The heuristic-neutralized-to-chance construction (now used at both n_objects=3 and 4) looks
  like the most methodologically important tool this project built -- worth leading with if
  the project resumes.
