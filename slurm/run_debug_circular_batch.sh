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
# claude/2026_08/2026_08_25/TODO.md and scripts/debug_circular/run_cluster_batch.py.
#
# *** --time=12:00:00 is a WORST-CASE PLACEHOLDER, not a measured value ***
# (200 trials x the OLD reasoning-enabled ~200s/trial pacing + ~7min load
# ~= 11h10m -- see qwen38_cluster_handoff.md's Known Bug #7). This job runs
# with thinking DISABLED, so the real time should be far lower. Run
# slurm/profile_debug_circular.sh first and set --time (and
# run_cluster_batch.py's --chunk-size below) from its printed
# extrapolation before submitting this for the real sweep. This job
# resumes safely if re-submitted (skips trials that already have a
# cluster_response.json), so an initial too-generous --time is still a
# safe way to make progress even before the profiling numbers are in.
#
# Assumes trial data (data/debug_circular/<run_name>/trials/) plus
# scripts/ have been rsynced to /mnt/cup/people/km3199/finst-video-model/
# (see run_cluster_batch.py's docstring for the exact commands). Set
# TRIALS_ROOT below to match the run_name used when the stimuli were
# generated.

TRIALS_ROOT="/mnt/cup/people/km3199/finst-video-model/data/debug_circular/qwen3.8-27b-nothink/trials"

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

echo "=== Running debug_circular batch job (trials-root=$TRIALS_ROOT) ==="
python3 /mnt/cup/people/km3199/finst-video-model/scripts/debug_circular/run_cluster_batch.py \
    --trials-root "$TRIALS_ROOT" \
    --chunk-size 20

echo "=== Disk check after run ==="
df -h "$HOME"

rm -rf "$TMPDIR"
