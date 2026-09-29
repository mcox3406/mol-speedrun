#!/bin/bash
#SBATCH --job-name=mol-v0-audit
#SBATCH --partition=mit_normal_gpu
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --output=logs/audit-%j.out
#SBATCH --error=logs/audit-%j.err
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
export XDG_CACHE_HOME="$SLURM_SUBMIT_DIR/cache/audit"
export TMPDIR="$SLURM_SUBMIT_DIR/tmp/audit"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
/home/mcox340/.conda/envs/sevennet-reewc/bin/python -u audit.py --data data --run results/reference-20260929.json --output results/audit.json
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
