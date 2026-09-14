#!/bin/bash
#SBATCH --job-name=gemma_pylyshyn_local_n4_chance_mp4
#SBATCH --partition=all
#SBATCH --gres=gpu:A100-40G:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=slurm/logs/gemma_pylyshyn_local_n4_chance_mp4_%j.out

# Runs the local-gemma-4-31b-it pylyshyn n_objects=4 "heuristic-neutral"
# replication, mp4-input variant: same 100 trials as
# run_gemma_pylyshyn_local_n4_chance.sh (50 target/50 distractor, heuristic
# at chance by construction), fed video.mp4 directly instead of video.npy.
# See claude/2026_09/2026_09_14/TODO.md's Task 2 and
# scripts/pylyshyn/run_gemma_local_batch_mp4.py.
#
# *** --time=00:45:00 above is a PLACEHOLDER, unprofiled. *** This job
# resumes safely if re-submitted (skips trials that already have a
# cluster_response_mp4.json), so an initial too-generous --time is still a
# safe way to make progress.
#
# Assumes trial data (data/pylyshyn/gemma_local_n4_heuristic_chance/trials/,
# including video.mp4) has been rsynced to
# /mnt/cup/people/km3199/finst-video-model/, and scripts/ are up to date
# via git pull on the cluster-side clone.

TRIALS_ROOT="/mnt/cup/people/km3199/finst-video-model/data/pylyshyn/gemma_local_n4_heuristic_chance/trials"

module load cudatoolkit/12.4.1

# Makes torchcodec's mp4 decoding work -- points at the micromamba-installed
# ffmpeg build's shared libs. See
# claude/skills/cluster/ffmpeg/test_torchcodec_with_module.sh.
export LD_LIBRARY_PATH="$HOME/micromamba/envs/ffmpeg-env/lib:$LD_LIBRARY_PATH"

source "$HOME/local-llm/.venv/bin/activate"

echo "=== Disk check before run ==="
df -h "$HOME"

echo "=== Running gemma pylyshyn local batch, n4 heuristic-chance mp4 variant (trials-root=$TRIALS_ROOT) ==="
python3 /mnt/cup/people/km3199/finst-video-model/scripts/pylyshyn/run_gemma_local_batch_mp4.py \
    --trials-root "$TRIALS_ROOT"

echo "=== Disk check after run ==="
df -h "$HOME"
