# Engaging GPU calibration

**Completed on Engaging:** one NVIDIA L40S, Slurm job `24302196`, 4m35s total job runtime. The 1,903,491-parameter SMILES transformer used 64 epochs selected on the inner chemical holdout, then a fresh all-training refit. This is one-seed calibration evidence, not an official speedrun record.

| Validation metric | Fingerprint MLP, 3-seed mean | SMILES transformer, 1 seed |
|---|---:|---:|
| S1 MAE (eV) | 0.269 | **0.250** |
| T1 MAE (eV) | 0.240 | **0.213** |
| Gap MAE (eV) | 0.171 | **0.150** |
| Log-intensity MAE | 0.325 | **0.308** |
| Raw-f MAE on f≥0.1 | 0.188 | **0.149** |

Same 75,280/16,257 chemical train/validation split as the CPU study. The fingerprint baselines fit separate energy and intensity networks; the transformer fits jointly, so this comparison does not isolate representation or architecture. No significance claim from one GPU seed.

Preprocessing: **1.37s**. Training-only tuning: **143.79s**. Fresh refit, including model/optimizer initialization: **123.42s**. Original final evaluation: **0.08s**. CUDA context/import startup and checkpoint serialization are outside these individual clocks; the Slurm elapsed time includes them. Peak allocated GPU memory: **615MB**. One final checkpoint is also backed up on the external drive under `qcdge/gpu-calibration/`.

**Implication:** this pilot is too small to establish a 10–60-minute challenge. Use the full eligible cohort and calibrate tighter, independently enforced energy/intensity requirements before freezing a leaderboard. The measured fit is about two minutes, not ten; do not manufacture difficulty with a minimum runtime.

The original training source is preserved in git commit `538dc62`. `results/plain-24302196.json` records the training run. `results/plain-24302196-evaluation.json` is the authoritative final validation: a fixed-checkpoint reevaluation (job `24304930`) corrects an f=0.1 boundary issue caused by transforming and reconstructing labels in float32. No retraining or model selection occurred. Energy/log-intensity metrics are identical; the bright subset correctly contains 2,202 molecules. Source, data, checkpoint and evaluation hashes are recorded and checked.

- One L40S via `mit_normal_gpu`, four CPUs, 16GB RAM, 30-minute Slurm limit.
- Scratch: `/home/mcox340/orcd/scratch/mol-speedrun-calibration-20260929` (resolves under `/orcd/scratch/orcd/008/mcox340`). Initial staging: 8.3MB, six files; after both completed jobs: **16MB, fourteen files**, including the single checkpoint. Only CSVs, their checksum manifest, chemical split IDs, Python source and a batch script are staged. The 48GB archive and feature caches stay off-cluster.
- Reuses `/home/mcox340/.conda/envs/sevennet-reewc/bin/python`, PyTorch 2.5.1+cu124. No environment installation, modification, or copy. Python bytecode is disabled and caches/temp paths point into the task's scratch directory.
- Completed training job: `24302196`; diagnostic reevaluation: `24304930`. Both completed successfully.

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
