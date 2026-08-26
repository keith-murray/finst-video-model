#!/bin/bash
#SBATCH --job-name=debug_circular_profile
#SBATCH --partition=all
#SBATCH --gres=gpu:A100-40G:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/debug_circular_profile_%j.out

# Profiles per-trial Qwen3.8-27B inference time on the debug_circular
# stimulus (see claude/2026_08/2026_08_25/TODO.md,
# claude/2026_08/2026_08_26/TODO.md, and
# scripts/debug_circular/profile_cluster_timing.py). Run this BEFORE
# submitting run_debug_circular_batch.sh, to pick a real --time and
# --chunk-size instead of guessing.
#
# Usage:
#   sbatch slurm/profile_debug_circular.sh              # nothink baseline
#   sbatch slurm/profile_debug_circular.sh low           # reasoning_effort=low
#   sbatch slurm/profile_debug_circular.sh medium preserve   # + eyeball reasoning text
#
# $1 = reasoning effort (low|medium|xhigh), omit for the nothink baseline.
# $2 = literal "preserve" to also pass --preserve-thinking (only meaningful
#      alongside $1) -- use this for the couple of eyeball-the-reasoning
#      profile runs the TODO asks for, not for every profile run.
#
# Assumes the repo's scripts/ directory, plus at least a few generated
# debug_circular trials (data/debug_circular/debug_circular_samples/),
# have already been rsynced to /mnt/cup/people/km3199/finst-video-model/
# (see the rsync commands documented in run_cluster_batch.py's docstring).
# --time=02:00:00 is a generous placeholder covering both the nothink case
# (30min would suffice) and a reasoning-enabled profile (default
# --chunk-sizes 5 10 16 means 31 total generations at up to ~200s/trial
# worst-case per the cluster handoff doc's Known Bug #7, ~103min, plus
# ~7min model load) -- override with `sbatch --time=...` if a specific
# effort level's profiling run needs more.

REASONING_EFFORT="${1:-}"
PROFILE_ARGS=()
if [[ -n "$REASONING_EFFORT" ]]; then
    PROFILE_ARGS+=(--reasoning-effort "$REASONING_EFFORT" --max-tokens 8192)
    if [[ "${2:-}" == "preserve" ]]; then
        PROFILE_ARGS+=(--preserve-thinking)
    fi
fi

module load cudatoolkit/12.4.1

export TMPDIR=/tmp/$USER/$SLURM_JOB_ID
mkdir -p "$TMPDIR"
export VLLM_CACHE_ROOT="$TMPDIR/vllm_cache"
export TRITON_CACHE_DIR="$TMPDIR/triton_cache"
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/inductor_cache"
export XDG_CACHE_HOME="$TMPDIR/xdg_cache"
export VLLM_USE_FLASHINFER_SAMPLER=0

source "$HOME/qwen38/.venv/bin/activate"

echo "=== Disk check before run ==="
df -h "$HOME"

echo "=== Running debug_circular profiling job (reasoning_effort=${REASONING_EFFORT:-nothink}) ==="
python3 /mnt/cup/people/km3199/finst-video-model/scripts/debug_circular/profile_cluster_timing.py \
    "${PROFILE_ARGS[@]}"

echo "=== Disk check after run ==="
df -h "$HOME"

rm -rf "$TMPDIR"
