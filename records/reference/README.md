# v0 reference measurement

All three prescribed seeds pass every target with the released reference trainer. Times include data processing, model initialization, training, all validation checks, and prediction/checkpoint exports:

| Seed | Inclusive time | Epochs |
|---|---:|---:|
| 20260929 | 365.07 s | 52 |
| 20260930 | 300.42 s | 43 |
| 20260931 | 291.11 s | 35 |

**Median: 300.42 seconds (5.01 minutes).** One L40S, four CPU cores, 16 GiB host allocation. This is the initial reference, not a claim of a statistically significant improvement over an incumbent. Training-only tuning from earlier calibration is disclosed in `experiments/full_cohort/`; it is not included in these fresh runs.

The recipe and all five targets were frozen before the three runs. The first seed's checkpoint is used for the supplementary rare-family audit after the full panel finished. Audit metrics are report-only and were not used to select the recipe.

Audit MAE: S1 **0.242 eV**, T1 **0.205 eV**, gap **0.152 eV**, transformed f **0.289**, and strong-transition raw f **0.082** (5,723 molecules). Energy errors rise on rare families; do not interpret validation speed as universal extrapolation accuracy.

Reproduce from **v0.1.0**, following the main README. Original reports retain staging revision `61d5421`; the released `race.py`, `model.py`, and `evaluate.py` match their recorded SHA256s exactly. Subsequent release changes corrected archive packaging/downloader metadata and added documentation and validation tools, without changing the executed recipe.

The [reference artifacts](https://github.com/mcox3406/mol-speedrun/releases/download/v0.1.0/reference-artifacts.tar.gz) contain predictions, checkpoints, reports, and logs. Verify the archive against `artifacts.json`, extract it, then run:

```sh
python validate_submission.py \
  --runs /path/reference/reference-20260929.json /path/reference/reference-20260930.json /path/reference/reference-20260931.json \
  --audit /path/reference/audit.json
```

`summary.json` contains independently recomputed validation and audit scores. The automatic checker says reproduction pending by design: the dashboard labels this maintainer-executed baseline **reference**, not an independently reproduced community record.
