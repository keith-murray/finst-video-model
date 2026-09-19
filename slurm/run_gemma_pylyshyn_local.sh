#!/bin/bash
#SBATCH --job-name=gemma_pylyshyn_local
#SBATCH --partition=all
#SBATCH --gres=gpu:A100-40G:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/gemma_pylyshyn_local_%j.out

# Runs the local-gemma-4-31b-it pylyshyn n_objects={3,4,5} sweep: 60 trials
# (20 seeds/n_objects, 10 match/10 non-match, fast redirect/fast speed,
# native only) at a single fixed condition -- num_frames=32, HF
# "recommended" sampling -- loading the model once for the whole sweep. See
# claude/2026_09/2026_09_10/TODO.md and
# scripts/pylyshyn/run_gemma_local_batch.py.
#
# *** --time=00:30:00 above is a PLACEHOLDER, unprofiled. *** Only 60 trials
# (vs. 2026-09-09's 560-call debug_circular sweep), so this should be
# generous, but this job resumes safely if re-submitted (skips trials that
# already have a cluster_response.json), so an initial too-generous --time
# is still a safe way to make progress even without profiling first.
#
# Assumes trial data (data/pylyshyn/gemma_local_nobjects_sweep/trials/) plus
# scripts/ have been rsynced to /mnt/cup/people/<netid>/finst-video-model/
# (see run_gemma_local_batch.py's docstring for the exact commands).

TRIALS_ROOT="/mnt/cup/people/<netid>/finst-video-model/data/pylyshyn/gemma_local_nobjects_sweep/trials"

module load cudatoolkit/12.4.1

source "$HOME/local-llm/.venv/bin/activate"

echo "=== Disk check before run ==="
df -h "$HOME"

echo "=== Running gemma pylyshyn local batch (trials-root=$TRIALS_ROOT) ==="
python3 /mnt/cup/people/<netid>/finst-video-model/scripts/pylyshyn/run_gemma_local_batch.py \
    --trials-root "$TRIALS_ROOT"

echo "=== Disk check after run ==="
df -h "$HOME"
