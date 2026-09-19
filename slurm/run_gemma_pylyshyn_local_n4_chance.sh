#!/bin/bash
#SBATCH --job-name=gemma_pylyshyn_local_n4_chance
#SBATCH --partition=all
#SBATCH --gres=gpu:A100-40G:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=slurm/logs/gemma_pylyshyn_local_n4_chance_%j.out

# Runs the local-gemma-4-31b-it pylyshyn n_objects=4 "heuristic-neutral"
# replication, npy-input variant: 100 trials (50 target/50 distractor,
# heuristic at chance by construction -- see
# scripts/pylyshyn/generate_gemma_local_n4_heuristic_chance_stimuli.py),
# fast redirect/fast speed, native only. Sibling of
# run_gemma_pylyshyn_local.sh, just retargeted at the new trials root with
# a larger time budget for 100 vs. 60 trials. See
# claude/2026_09/2026_09_14/TODO.md's Task 2 and
# scripts/pylyshyn/run_gemma_local_batch.py.
#
# *** --time=00:45:00 above is a PLACEHOLDER, unprofiled. *** This job
# resumes safely if re-submitted (skips trials that already have a
# cluster_response.json), so an initial too-generous --time is still a
# safe way to make progress.
#
# Assumes trial data (data/pylyshyn/gemma_local_n4_heuristic_chance/trials/)
# has been rsynced to /mnt/cup/people/<netid>/finst-video-model/, and
# scripts/ are up to date via git pull on the cluster-side clone.

TRIALS_ROOT="/mnt/cup/people/<netid>/finst-video-model/data/pylyshyn/gemma_local_n4_heuristic_chance/trials"

module load cudatoolkit/12.4.1

source "$HOME/local-llm/.venv/bin/activate"

echo "=== Disk check before run ==="
df -h "$HOME"

echo "=== Running gemma pylyshyn local batch, n4 heuristic-chance npy variant (trials-root=$TRIALS_ROOT) ==="
python3 /mnt/cup/people/<netid>/finst-video-model/scripts/pylyshyn/run_gemma_local_batch.py \
    --trials-root "$TRIALS_ROOT"

echo "=== Disk check after run ==="
df -h "$HOME"
