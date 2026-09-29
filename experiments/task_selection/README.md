# Quantum task selection — development study

**Recommendation:** PCQM4Mv2 HOMO–LUMO gap, measured in eV, with a chemically grouped holdout and a clearly defined neutral organic domain. ESOL remains a pipeline smoke test. The GPU time-to-quality threshold is **not frozen**.

Read [REPORT.md](REPORT.md) for the actual results and proposed protocol. No random-molecule or official-CID-split result is used in this recommendation. Assignment of whole chemical groups is seeded; sampling within the resulting training partition is permitted for learning curves.

## Reproduce

Use the repository's Python 3.12 environment and pinned `requirements.txt`. From this directory:

```sh
python download.py
# The 210k-row screening pool samples only publicly labeled OGB train/valid rows.
python screen.py --dataset pcqm --split scaffold --train-size 200000
python audit_groups.py
python domain.py
python audit_duplicates.py
python screen.py --dataset pcqm --split chemical --train-size 200000 --closed-shell --skip-forest
python fingerprint_mlp.py --train-size 200000 --split chemical --closed-shell --epochs 25
python -m unittest discover -s . -p 'test_selection.py'
```

`--train-size 200000` chooses 200k original OGB training rows plus 10k original OGB validation rows **as a development pool**, before making entirely new chemical splits. It does not mean 200k final training molecules. Hidden OGB test labels are never accessed. The final filtered counts are 156,627 train / 36,122 development validation. No final test set is claimed in this exploratory study.

For the unfiltered matched-pool learning curve:

```sh
python fingerprint_mlp.py --train-size 200000 --split chemical --fit-size 50000 --epochs 25
python fingerprint_mlp.py --train-size 200000 --split chemical --epochs 25
```

For the QM9 diagnostic:

```sh
python screen.py --dataset qm9 --split scaffold
```

## Method

- Data URLs and SHA-256 digests: `sources.json`. Feature caches and source downloads are ignored by git. Per-split source-row indices are retained in `results/*-indices.json`. The RDKit version and invalid/duplicate counts are recorded in each screening result.
- Canonical isomeric SMILES identifies duplicates; the first source row is retained. No tautomerization, neutralization or salt stripping modifies the quantum-chemistry labels. Duplicate label discordance is reported separately.
- `scaffold`: non-chiral typed Bemis–Murcko scaffold. `chemical`: typed scaffold for cyclic molecules, untyped full heavy-atom topology for acyclic molecules. `generic`: untyped ring scaffold plus the same acyclic rule; audited as an alternative, not selected for the primary study.
- Entire chemical groups go into either train or validation, with 20% of **groups**, not necessarily molecules, held out. Inner hyperparameter/epoch selection also holds out whole groups. Consequently the inner molecule counts can be uneven; exact counts are recorded. The outer validation is never used to choose model hyperparameters or epochs.
- Features: 2,048 Morgan radius-2 bits, 30 cheap descriptors/counts, or atom counts alone. No quantum properties, conformers or target-correlated source columns are input features. QM9 gap/total energy convert Hartree to eV; dipole remains Debye.
- Ridge alpha, descriptor boosting regularization and forest leaf size use small prespecified grids. Fingerprint MLP: 565,249 parameters, 256/128 hidden units, AdamW, batch size 512, 25-epoch inner search; selected epoch count is refitted from scratch on all training molecules.
- Timings are CPU feasibility checks, with shared cached features and sometimes concurrent processes. They are **not eligible speedrun times**. The final MLP validation score is from its prespecified final epoch, not the best outer-validation checkpoint.
- MAE bootstrap intervals resample molecules conditional on one fitted model/split. They are not scaffold-cluster confidence intervals and do not measure training-seed uncertainty. Treat them as descriptive only.

The study developed incrementally: older exploratory files use the earlier Cholesky ridge solver (which emitted conditioning warnings); the final closed-shell results use SVD. Early ESOL/lipophilicity forests used 128 trees and 20% of features; quantum screening uses 64 trees with square-root feature sampling. Exact primary commands above reproduce the final protocol; the secondary files document the earlier diagnostics. The initial molecule-random/CID fits were abandoned and are excluded from this directory.
