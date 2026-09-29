#!/bin/bash
#SBATCH --job-name=mol-full-gpu
#SBATCH --partition=mit_normal_gpu
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:50:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
export XDG_CACHE_HOME="$SLURM_SUBMIT_DIR/cache"
export CUDA_CACHE_PATH="$SLURM_SUBMIT_DIR/cache/cuda"
export TMPDIR="$SLURM_SUBMIT_DIR/tmp"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
/home/mcox340/.conda/envs/sevennet-reewc/bin/python -u calibrate.py --data data --split split.json --output "results/plain-${SLURM_JOB_ID}.json" --tune-seconds 1200 --refit-seconds 1500
