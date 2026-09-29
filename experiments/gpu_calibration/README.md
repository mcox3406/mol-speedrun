# Engaging GPU calibration

Compact supervised SMILES-transformer calibration on the existing QCDGE development split. This is a calibration experiment, not an official speedrun record.

- One L40S via `mit_normal_gpu`, four CPUs, 16GB RAM, 30-minute Slurm limit.
- Scratch: `/home/mcox340/orcd/scratch/mol-speedrun-calibration-20260929` (resolves under `/orcd/scratch/orcd/008/mcox340`). Initial staging: 8.3MB, six files. Only CSVs, their checksum manifest, chemical split IDs, Python source and a batch script are staged. The 48GB archive and feature caches stay off-cluster.
- Reuses `/home/mcox340/.conda/envs/sevennet-reewc/bin/python`, PyTorch 2.5.1+cu124. No environment installation, modification, or copy. Python bytecode is disabled and caches/temp paths point into the task's scratch directory.
- First submission: Slurm job `24302196`. Check scheduler/result files for completion; submission is not a successful calibration result.

## Model and selection

ASCII-character encoder, four bidirectional transformer blocks, width192, six heads, masked mean pooling, three outputs: S1, T1, transformed oscillator strength. Gap is S1−T1. AdamW, learning rate3e−4, batch256, dropout0.1, bfloat16 on GPU. Loss averages training-standard-deviation-normalized absolute errors on S1, T1, gap and log-intensity.

Select epochs on the existing **inner chemical holdout**, with a ten-minute/200-epoch tuning cap and patience20. Reinitialize and fit all training molecules for that epoch count, with a separate twelve-minute safety cap. Evaluate the outer validation set once. The run reports an incomplete status if the refit budget prevents completion of the selected epoch count. Selection never uses outer-validation checkpoints.

Timing starts after imports and CUDA context initialization. Preprocessing, tuning, fresh refit (including model/optimizer initialization), and final evaluation are reported separately. No compilation or pretraining. These clocks are explicit calibration measurements, not the final proposed speedrun rules. One training seed and one chemical partition do not establish statistical significance.

## Run

Stage `train.csv`, `validation.csv`, and `manifest.json` under `data/`; copy the matching `qcdge-100000-indices.json` to `split.json`. The trainer checks source and split digests before training. Place `calibrate.py` and `calibrate.sbatch` in the scratch working directory; create `logs/` and `results/`; run `sbatch calibrate.sbatch` there. The environment path in the batch script is specific to this account.

Outputs are one JSON report, one final checkpoint, and Slurm stdout/stderr. No per-epoch checkpoints or prediction dumps. Copy results back before relying on scratch for retention.

Local validation:

```sh
python -m unittest discover -s experiments/gpu_calibration -p 'test_*.py'
```

The CPU smoke path uses 128 rows per partition and a one-epoch tiny model; its output is explicitly marked `cpu_smoke_only` and is not a benchmark score. Padding invariance and finite gradient propagation are tested.

Cluster resource choices follow [ORCD's GPU resource guidance](https://orcd-docs.mit.edu/running-jobs/requesting-resources/); scratch location follows the [filesystem documentation](https://orcd-docs.mit.edu/filesystems-file-transfer/filesystems/).
