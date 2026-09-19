# Do VLMs have FINST-like object tracking mechanisms?

Video language models (VLMs) have made large strides over the past two years. VLMs can now digest hour-long videos and provide relevant descriptions of the video. This raises the question, "What are the emergent mechanisms in VLMs that allow them to reason about videos?" To constrain this question, we lean on the work of Zenon Pylyshyn to ask, "How do VLMs track multiple objects?"

Pylyshyn's work proposes an index-like mechanism that agents assign to objects moving in the environment, termed (F)ingers of (INST)antiation (FINST). Do VLMs have a FINST-like mechanism to track objects?

**The project is currently paused** (see [Status](#status) below), but the code, stimuli, and results are published here for reproducibility.

## Approach

The project generates synthetic multiple-object-tracking (MOT) videos — several identical objects move on screen, a subset are cued at the start, then all objects become visually identical and move for a tracking phase, and the VLM is asked which objects were cued. This is the classic Pylyshyn/MOT paradigm, adapted to a video-question-answering format so any video-capable VLM (via OpenRouter or a locally-hosted model) can be tested with no special interface.

Three stimulus families live under `src/finst_video_model/`:

- **`pylyshyn`** — the canonical task: objects move on random walks, a subset are cued then tracked, the model reports which were cued.
- **`debug_circular`** — objects move along interpolable circular paths (rather than a random walk), used for cleaner mechanistic follow-up once the random-walk task turned out to be confounded (see Status).
- **`smooth_pursuit`** — an earlier/simpler stimulus variant.

## Repo layout

```
src/finst_video_model/   library code: stimulus generation (physics/config/stimulus_gen per task) and the OpenRouter/vLLM query client (vlm_client.py)
scripts/                 experiment entry points: generate stimuli, run model sweeps, aggregate, plot -- organized by stimulus family
results/                 aggregate CSVs and plots checked into git (per-trial raw videos/frames are gitignored, kept locally under data/)
slurm/                   SLURM batch scripts for running local models (e.g. Gemma, Qwen) on a GPU cluster
claude/                  dated research-session logs (SUMMARY.md/TODO.md per session) -- the actual research diary for this project, kept for transparency
```

## Setup

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env   # fill in OPENROUTER_API_KEY for any OpenRouter-backed script
```

Most experiments are run directly as scripts, e.g.:

```bash
uv run python scripts/pylyshyn/generate_samples.py ...
uv run python scripts/pylyshyn/run_pylyshyn_trial.py ...
```

See a given stimulus family's `scripts/<family>/` directory and the dated notes under `claude/` for the exact invocation used to produce a given `results/` entry.

The `slurm/` scripts assume access to a Princeton-style SLURM cluster with paths like `/mnt/cup/people/<netid>/...` and `/scratch/<netid>/...` — these are left as reference for adapting to your own cluster, not runnable as-is.

## Status

As of the last session (2026-09-16), the project is **paused indefinitely**. Key findings along the way:

- Several VLMs score at chance (d′ ≈ 0) on the tracking task without any reasoning/chain-of-thought; enabling reasoning rescues most (but not all) models to ~88–93% on a hardened variant.
- A trivial nearest-neighbor position heuristic scores about as well as real models on the standard random-walk task, meaning it doesn't cleanly separate genuine object tracking from position-guessing. The `debug_circular` stimulus was built to get a cleaner signal, and shows a real angle-dependent capacity curve under that heuristic control.
- A persistent accuracy gap between locally-hosted and OpenRouter-hosted copies of the same model was traced and several candidate causes (sampling, prompting, reasoning, frame selection, numeric precision) were ruled out; root cause remains open.
- The intended end goal was mechanistic interpretability of tracking capacity (not benchmarking), so reasoning-trace models were deprioritized late in the project since chain-of-thought obscures the mechanism of interest.

See `claude/2026_09/2026_09_16/SUMMARY.md` and the dated session logs under `claude/` for full detail on each experiment.

## License

MIT — see [LICENSE](LICENSE).
