#!/bin/bash
#SBATCH --job-name=debug_circular_batch
#SBATCH --partition=all
#SBATCH --gres=gpu:A100-40G:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/debug_circular_batch_%j.out

# Runs the full debug_circular sweep (200 trials: rotation_deg in
# {20,40,60,80,100} x clockwise in {True,False} x probe_on_target in
# {True,False} x 10 seeds) against the locally-hosted Qwen3.8-27B. See
# claude/2026_08/2026_08_25/TODO.md, claude/2026_08/2026_08_26/TODO.md, and
# scripts/debug_circular/run_cluster_batch.py.
#
# Usage:
#   sbatch --time=<measured> slurm/run_debug_circular_batch.sh              # nothink baseline
#   sbatch --time=<measured> slurm/run_debug_circular_batch.sh low          # reasoning_effort=low
#   sbatch --time=<measured> slurm/run_debug_circular_batch.sh medium
#   sbatch --time=<measured> slurm/run_debug_circular_batch.sh xhigh
#
# $1 = reasoning effort (low|medium|xhigh), omit for the nothink baseline.
# All three reasoning levels point at the SAME already-populated
# qwen3.8-27b-nothink trials dir (reusing the existing 200 videos rather
# than regenerating/duplicating ~84MB/trial of raw frame data); each level
# writes its own cluster_response_<effort>.json alongside the existing
# plain cluster_response.json, so all four conditions can coexist under
# one trials dir without collisions.
#
# *** --time=01:00:00 is a PLACEHOLDER for the nothink case only ***
# (measured ~18min total incl. model load for nothink -- see
# claude/2026_08/2026_08_25/SUMMARY.md). For any --reasoning-effort level,
# this WILL be far too short -- thinking-enabled generation can take up to
# ~200s/trial per qwen38_cluster_handoff.md's Known Bug #7 (200 trials x
# 200s + ~7min load ~= 11h10m worst case). Always run
# slurm/profile_debug_circular.sh for that effort level FIRST and pass its
# printed extrapolation via `sbatch --time=...` (overrides this file's
# directive) before submitting a reasoning-enabled batch. This job resumes
# safely if re-submitted (skips trials that already have the relevant
# response file), so an initial too-generous --time is still a safe way to
# make progress even before the profiling numbers are in.
#
# Assumes trial data (data/debug_circular/<run_name>/trials/) plus
# scripts/ have been rsynced to /mnt/cup/people/km3199/finst-video-model/
# (see run_cluster_batch.py's docstring for the exact commands).

REASONING_EFFORT="${1:-}"
TRIALS_ROOT="/mnt/cup/people/km3199/finst-video-model/data/debug_circular/qwen3.8-27b-nothink/trials"

BATCH_ARGS=(--trials-root "$TRIALS_ROOT" --chunk-size 20)
if [[ -n "$REASONING_EFFORT" ]]; then
    BATCH_ARGS+=(--reasoning-effort "$REASONING_EFFORT" --max-tokens 8192)
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

echo "=== Running debug_circular batch job (trials-root=$TRIALS_ROOT, reasoning_effort=${REASONING_EFFORT:-nothink}) ==="
python3 /mnt/cup/people/km3199/finst-video-model/scripts/debug_circular/run_cluster_batch.py \
    "${BATCH_ARGS[@]}"

echo "=== Disk check after run ==="
df -h "$HOME"

rm -rf "$TMPDIR"
