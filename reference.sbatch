#!/bin/bash
#SBATCH --job-name=mol-v0-reference
#SBATCH --partition=mit_normal_gpu
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:05:00
#SBATCH --array=0-2%1
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
export XDG_CACHE_HOME="$SLURM_SUBMIT_DIR/cache/$SLURM_ARRAY_TASK_ID"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
export TMPDIR="$SLURM_SUBMIT_DIR/tmp/$SLURM_ARRAY_TASK_ID"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
seed=$((20260929 + SLURM_ARRAY_TASK_ID))
/home/mcox340/.conda/envs/sevennet-reewc/bin/python -u race.py --data data --seed "$seed" --output "results/reference-$seed.json"
