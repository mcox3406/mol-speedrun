# Full-cohort development calibration

Historical calibration record. The released competition contract is now in [RULES.md](../../RULES.md); its timed reference results are separate from these experiments.

The full metadata-indexed QCDGE extraction is complete. This is calibration, not an official speedrun protocol or a frozen test benchmark.

| Fold | Molecules | Chemical groups |
|---|---:|---:|
| Training | 308,402 | 10,734 |
| Validation | 64,549 | 2,684 |
| Unused reserve | 31,148 | 23,212 |

Training contains an inner chemical split of 261,923 / 46,479 for model selection. All pilot family assignments are preserved as their families expand; every previously unseen family goes into the reserve. There is no random molecule split. Assertions check disjoint group IDs, typed scaffolds (generic topology for acyclic molecules), Standard InChI connectivity families, and canonical isomeric SMILES across train/validation/reserve and across the inner folds. These checks do not imply all forms of chemical similarity are eliminated.

**The reserve is dominated by rare families (maximum 18 molecules per group).** It is neither distribution-matched nor yet certified as the final test set. Its labels do not enter training, calibration scores, or the blog EDA. Standard physical-validity checks were applied to all folds.

Of 404,188 unique structurally eligible molecules, 89 were excluded: 54 representatives of repeated canonical structures (all duplicate pairs quarantined, independent of disagreement size), 29 nonphysical excitation records, and six unavailable/malformed records. The complete metadata extraction has 441,404 entries and eight parsing failures; two failed entries fall outside the structural domain. The archive/paper/metadata count discrepancy remains unresolved: see [QCDGE audit](../task_selection/QCDGE.md). The source domain is small organic molecules; training heavy-atom 5th/50th/95th percentiles are 8/10/10. This is not a drug-size extrapolation benchmark.

## Reproduce preparation

Keep the large archive on the external drive. The data directory contains the verified source HDF5/CSV, the structure feature cache, and extraction CSV/audits produced by the earlier task-selection scripts.

```sh
python experiments/task_selection/extract_qcdge.py --directory "$QCDGE_DIR" --sample 0 --workers 4
python experiments/full_cohort/prepare.py \
  --directory "$QCDGE_DIR" \
  --pilot experiments/task_selection/results/qcdge-100000-indices.json \
  --output "$QCDGE_DIR/full-cohort"
```

The output includes immutable fold lists, checksummed train/validation CSVs, a compact compressed fingerprint/descriptor cache, the reserve IDs, and an audit. `matched-pilot-validation.json` lists 16,254 retained pilot validation molecules for a future paired comparison; the full validation score alone must not be presented as an apples-to-apples improvement over the pilot score. The fold SHA256 is `02653cca9d24c99a82dfcefce545ed63be0e857041f348b4b675500d03b84608`.

## Engaging calibration

Scratch directory: `/home/mcox340/orcd/scratch/mol-speedrun-full-20260929`. Initial footprint **58 MB / 12 files**, with no copied environment or source archive. Both jobs reuse `/home/mcox340/.conda/envs/sevennet-reewc/bin/python` and place caches/temp files in scratch.

- GPU job **24310980**: one L40S, four CPU cores, 16 GB RAM; identical 1.90M-parameter bidirectional SMILES model to the pilot. Inner chemical tuning up to 1,200 seconds; fresh fit up to 1,500 seconds; Slurm limit 50 minutes. One training seed. `gpu.sbatch` calls `../gpu_calibration/calibrate.py`, staged as `calibrate.py`.
- CPU job **24310981**: four CPU cores, 32 GB RAM, two-hour limit. Training medians, atom-count ridge, Morgan ridge, a 64-tree fingerprint/descriptor forest, and a fingerprint/descriptor MLP with three seeds. Tune only on the inner chemical holdout and refit on all training molecules.

These CPU controls fit S1, T1, and transformed oscillator strength jointly, unlike the pilot's separately fitted energy and intensity models. Gap predictions are differences. The neural loss averages training-standard-deviation-normalized MAEs for S1, T1, gap, and `log10(1+f/0.001)`. Raw oscillator MAE and a strong-transition (`f >= 0.1`) slice guard against hiding errors through the transform. The transform and thresholds remain provisional.

GPU timing separates CSV loading/tokenization, inner tuning, fresh fit, and evaluation; imports and CUDA context setup are excluded. CPU timings use cached features. Neither is an eligible competition record. Do not infer statistical superiority from one GPU seed. A chosen epoch count that cannot finish within the refit budget is explicitly flagged.

These experiments informed the frozen v0 targets. The release uses an inclusive clock and a three-seed reference panel, plus a separately labeled rare-family audit. Increasing dataset size alone does not establish useful difficulty.
