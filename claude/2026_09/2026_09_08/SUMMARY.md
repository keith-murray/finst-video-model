# Summary: measured gemma-4-31b-it's real frame-sampling budget and confirmed it's provider-independent

Full context/goals: `claude/2026_09/2026_09_08/TODO.md` -- both tasks below
grew directly out of that one open question ("how many frames does
gemma-4-31b-it actually sample?").

## 1. Direct measurement: hand-built 1-50 frame sweep (the TODO's ask)

Prior sessions established `gemma-4-31b-it` extracts a fixed-size frame
sample regardless of real video length (100/200/400-frame clips all gave
identical `prompt_tokens`), and a back-of-envelope estimate -- borrowing
Gemini's ~258-tokens/frame convention -- guessed the sample size at ~9
frames. That guess was never measured directly.

Built `scripts/debug_circular/run_gemma_frame_budget_sweep.py`: hand-
constructed minimal videos (a single white cross drifting a few pixels/frame
on a black 384x384 background, reusing `debug_circular.stimulus_gen`'s
`_draw_cross`/`write_mp4`) with exactly N real frames, sent each to
`google/gemma-4-31b-it` via OpenRouter, read `usage.prompt_tokens`. Started
at N=1-15 per the TODO; the result was still perfectly linear at N=15 with
no sign of a bend, so extended live to N=50 with the user's go-ahead after
flagging the surprise.

**Result** (100 calls, 0 errors, 2 reps/N, deterministic): `prompt_tokens`
fits `23 + 73 * min(n_frames, 32)` essentially exactly -- linear at ~73
tokens/frame through frame 32, then locks to exactly 2,359 tokens all the
way to 50. **The real budget is 32 real frames, not ~9** -- the earlier
guess's text-residual math was fine, only the assumed 258-tokens/frame rate
(borrowed from a different model) was wrong. `results/debug_circular/
frame_budget_sweep/prompt_tokens_by_frame_count.png`. Full derivation in
the `reference_gemma_fixed_frame_budget` memory.

Also revises the read on `gemma-4-31b-it`'s strong pylyshyn n_objects=5
result from a prior session: it's sampling ~32 of the ~100 real frames in
that stimulus, not ~9 -- a much less severe undersampling than previously
assumed.

## 2. Follow-up: is this a model property or a per-provider quirk?

Not part of the original TODO -- added after noticing OpenRouter's raw
response carries a top-level `"provider"` field, and an ad hoc check showed
Task 1's calls weren't confirmed to have landed on a single provider (one
unpinned call landed on `ModelRun`, not the assumed-default `DeepInfra`,
yet returned an identical token count). Extended `vlm_client.ask_about_video`
to also return this field (`usage["served_by_provider"]`) so any future run
can check.

Smoke-tested all 10 active OpenRouter endpoints for this model (n=5
frames each): 7 accept video input, 3 don't -- `friendli` (422), `sambanova`
(502), `cerebras` (404), each reproduced twice, so treated as not currently
serving video for this model rather than a flake.

New `scripts/debug_circular/run_gemma_frame_budget_provider_sweep.py`: the
same frame-count sweep (N=1-45), pinned per provider via
`provider={"only": [p], "allow_fallbacks": False}`, reusing Task 1's
already-generated videos. Estimated cost (~$0.13 for 315 calls, using each
provider's live pricing) before running, per project convention.

**Result: 6 of 7 providers (deepinfra, coreweave, crusoe, parasail,
together, modelrun) reproduce the identical `23 + 73 * min(n_frames, 32)`
curve**, agreeing to within 2 tokens at the plateau.
`results/debug_circular/frame_budget_provider_sweep/prompt_tokens_by_provider.png`
shows all 6 essentially on top of each other. `chutes` degraded into
sustained 503s for most n>=15 across two full retry passes (22/45 points
never returned), but its 23 successful points sit exactly on the same line.

**Conclusion**: since architecturally distinct serving stacks all reproduce
the same cap and rate, the 32-frame budget is almost certainly baked into
`gemma-4-31b-it`'s own video-processor/chat-template config, not something
each hosting provider independently implements.

## Process gotchas worth remembering

- **Off-by-one caught and fixed mid-write-up**: the plateau's last
  increasing step is 31->32 (+73); 32->33 is the first flat step -- so the
  cap is 32 frames, not 33. First drafted as "33" before the provider sweep
  independently reproduced the same 31/32/33 transition on 6 endpoints and
  forced a re-check.
- **A rare, reproducible CoreWeave usage-reporting glitch**: 2 of 90
  CoreWeave calls came back `prompt_tokens=16, cost=0` despite a normal,
  content-aware response -- not the model failing to see the video, a
  billing/telemetry undercount. Zero instances on the other 5 working
  providers (225 calls). Plain retry fixed both instantly. Written up in
  `reference_openrouter_provider_routing_gotchas` (item 5) in case it
  recurs and looks alarming again.
- Resume-by-existing-row logic (this project's standard batch-script
  pattern) doesn't distinguish a successful row from an errored one -- an
  errored `(provider, n_frames)` pair has to be explicitly deleted from
  `results.csv` before a re-run will retry it, same as every prior batch
  script here.

## Next steps for tomorrow: local gemma-4-31b-it hosting

User's plan: host `gemma-4-31b-it` locally (like the existing Qwen3.8-27B
cluster setup, `claude/skills/cluster/qwen38_cluster_handoff.md`) to test
whether the number of frames it actually receives affects task performance
directly, rather than only measuring `prompt_tokens` via OpenRouter. Now
that the real budget (32 frames) is pinned down, the natural next
experiment is controlling frame count explicitly at inference time and
checking accuracy on pylyshyn/debug_circular as it varies -- something
OpenRouter's hosted endpoint gives no lever for (it always resamples down to
its own fixed 32-frame budget internally, invisible to the caller). Local
hosting removes that black box the same way it did for Qwen3.8-27B.

Open questions to carry in:
- Whether gemma-4-31b-it's local processor/chat-template exposes a
  `max_frames`-style knob directly (analogous to Qwen's
  `enable_thinking`/`reasoning_effort` chat-template kwargs, see
  `reference_local_qwen_reasoning_effort`), or whether frame selection has
  to be done manually before the video ever reaches the processor.
  - If it does expose a knob: the model is 30.7B-dense (per OpenRouter's
    model description) vs. Qwen3.8-27B -- similar scale, so the existing
    2xA100 cluster setup should be adequate, but worth confirming a video
    checkpoint of this size loads cleanly before assuming so.
- Whether performance degrades gracefully or falls off a cliff as frame
  count drops well below 32 on pylyshyn (identity tracking) vs.
  debug_circular (continuous rotation) -- these have different sensitivity
  to temporal sparsity in principle, per the "10-frame stimulus" idea
  floated in `project-status-2026-09-01`.
- Whether feeding *more* than 32 real frames locally (no fixed budget to
  hit) improves accuracy at all, now that OpenRouter's ceiling is known to
  be 32 -- a natural, cheap add-on once local hosting is working.
