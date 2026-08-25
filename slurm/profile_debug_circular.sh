#!/bin/bash
#SBATCH --job-name=debug_circular_profile
#SBATCH --partition=all
#SBATCH --gres=gpu:A100-40G:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/debug_circular_profile_%j.out

# Profiles per-trial Qwen3.8-27B inference time on the debug_circular
# stimulus (see claude/2026_08/2026_08_25/TODO.md and
# scripts/debug_circular/profile_cluster_timing.py). Run this BEFORE
# submitting run_debug_circular_batch.sh, to pick a real --time and
# --chunk-size instead of guessing.
#
# Assumes the repo's scripts/ directory, plus at least a few generated
# debug_circular trials (data/debug_circular/debug_circular_samples/),
# have already been rsynced to /mnt/cup/people/km3199/finst-video-model/
# (see the rsync commands documented in run_cluster_batch.py's docstring).
# 30 minutes is a confident placeholder: model load is ~5-7min and this
# job only runs a handful of short (thinking-disabled) generations.

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

echo "=== Running debug_circular profiling job ==="
python3 /mnt/cup/people/km3199/finst-video-model/scripts/debug_circular/profile_cluster_timing.py

echo "=== Disk check after run ==="
df -h "$HOME"

rm -rf "$TMPDIR"
