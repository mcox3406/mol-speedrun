#!/bin/bash
#SBATCH --job-name=mol-qcdge-calibration
#SBATCH --partition=mit_normal_gpu
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export XDG_CACHE_HOME="$SLURM_SUBMIT_DIR/cache"
export TORCH_HOME="$SLURM_SUBMIT_DIR/cache/torch"
export CUDA_CACHE_PATH="$SLURM_SUBMIT_DIR/cache/cuda"
export TRITON_CACHE_DIR="$SLURM_SUBMIT_DIR/cache/triton"
export TMPDIR="$SLURM_SUBMIT_DIR/tmp"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
/home/mcox340/.conda/envs/sevennet-reewc/bin/python -u calibrate.py --data data --split split.json --output "results/plain-${SLURM_JOB_ID}.json" --tune-seconds 600 --refit-seconds 720
