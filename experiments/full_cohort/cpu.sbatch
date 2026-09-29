#!/bin/bash
#SBATCH --job-name=mol-full-controls
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
export XDG_CACHE_HOME="$SLURM_SUBMIT_DIR/cache"
export TMPDIR="$SLURM_SUBMIT_DIR/tmp"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
/home/mcox340/.conda/envs/sevennet-reewc/bin/python -u baselines.py --data baseline.npz --split split.json --output "results/cpu-${SLURM_JOB_ID}.json"
