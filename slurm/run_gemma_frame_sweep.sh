#!/bin/bash
#SBATCH --job-name=gemma_frame_sweep
#SBATCH --partition=all
#SBATCH --gres=gpu:A100-40G:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/gemma_frame_sweep_%j.out

# Runs the local-gemma-4-31b-it debug_circular frame-count sweep: 40 trials
# (rotation_deg=40, 20 match/20 non-match) x num_frames in
# {4,8,16,24,32,50,100} x sampling in {greedy,recommended} (560 total calls),
# loading the model once for the whole sweep. See
# claude/2026_09/2026_09_09/TODO.md and
# scripts/debug_circular/run_gemma_cluster_batch.py.
#
# *** --time=01:00:00 above is a PLACEHOLDER. ***
# Run scripts/debug_circular/profile_gemma_cluster_timing.py FIRST and pass
# its printed recommendation via `sbatch --time=<measured> ...` (overrides
# this file's directive) before submitting the real job -- same convention as
# slurm/run_debug_circular_batch.sh for the qwen3.8-27b pipeline. This job
# resumes safely if re-submitted (skips (trial, num_frames, sampling) triples
# that already have a cluster_response_nframes{N}_{sampling}.json), so an
# initial too-generous --time is still a safe way to make progress even
# before the profiling numbers are in.
#
# Assumes trial data (data/debug_circular/gemma_frame_sweep/trials/) plus
# scripts/ have been rsynced to /mnt/cup/people/km3199/finst-video-model/
# (see run_gemma_cluster_batch.py's docstring for the exact commands, and its
# "known unknowns" note about confirming this path against the gemma4
# environment's actual mount).

TRIALS_ROOT="/mnt/cup/people/km3199/finst-video-model/data/debug_circular/gemma_frame_sweep/trials"

module load cudatoolkit/12.4.1

source "$HOME/local-llm/.venv/bin/activate"

echo "=== Disk check before run ==="
df -h "$HOME"

echo "=== Running gemma frame sweep (trials-root=$TRIALS_ROOT) ==="
python3 /mnt/cup/people/km3199/finst-video-model/scripts/debug_circular/run_gemma_cluster_batch.py \
    --trials-root "$TRIALS_ROOT" \
    --num-frames 4 8 16 24 32 50 100 \
    --sampling greedy recommended

echo "=== Disk check after run ==="
df -h "$HOME"
