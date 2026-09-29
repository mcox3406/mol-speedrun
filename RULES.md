# QCDGE speedrun v0

**Minimize median elapsed time on one NVIDIA L40S. Every prescribed seed must meet every validation target within 3,600 seconds.** This is a small-molecule excited-state benchmark, not a claim about experimental accuracy or drug discovery.

## Fixed task

Use the checksummed [v0 dataset](DATA.md): 308,402 training and 64,549 validation structures. Inputs are the supplied 2D molecular structures. Predict S1 and T1 vertical excitation energies and S1 oscillator strength. The evaluator derives the singlet–triplet gap; it cannot be predicted independently to evade consistency.

| Mean absolute error | Maximum |
|---|---:|
| S1 | 0.180 eV |
| T1 | 0.150 eV |
| S1 − T1 | 0.115 eV |
| log₁₀(1 + f / 0.001) | 0.250 |
| Raw f, molecules with true f ≥ 0.1 | 0.120 |

All five conditions must hold separately for seeds **20260929, 20260930, 20260931**. Also report overall raw-f MAE. `protocol.json` and `evaluate.py` are the scoring authority. No averaging across targets or seeds to hide a failure. Prediction columns are `id,S1_eV,T1_eV,log_f`; preserve the supplied row order. Raw-f reconstruction is `max(0, 0.001 * (10**clip(log_f, -6, 6) - 1))`. Labels and scoring use float64; the threshold includes exactly f = 0.1.

## Allowed methods and selection

Any architecture or structure-derived representation is allowed. Train from scratch using only training labels. No pretrained weights, external molecules/labels, quantum calculations, cached label-bearing embeddings, ID-to-label tables, or use of validation/audit labels for gradients or feature construction. IDs are for joins only, never inputs. Chemical augmentation of training structures is allowed. Fit vocabularies, normalizers, and learned preprocessing on training data only.

The supplied inner chemical folds support development. Once a submission is frozen, use the same code, hyperparameters, and stopping rule for all three seeds. Full validation checks may stop training when all targets pass; no per-molecule validation feedback may update the model. Include every attempt with the frozen recipe, including failures. Do not replace seeds, average checkpoints across seeds, or adapt the recipe after seeing audit results. Development experiments are outside the race clock and must be disclosed in the PR.

## Hardware and clock

Allocate **one L40S (48 GB), at most four CPU threads/cores, and 16 GiB host RAM** per run. Other GPUs belong in exploratory results, not this ranking. No concurrent jobs on the same allocated GPU. The reference uses Python 3.11, PyTorch 2.5.1+cu124, and NumPy 1.26.4; record the complete software environment, driver, hardware, and allocated resources. Maintainers compare candidate and incumbent on the same hardware/software configuration.

Start a fresh process for each seed. Start the clock before reading the prepared data and before any submission-specific work: tokenization, graph/fingerprint construction, device transfers, model/optimizer construction, compilation/autotuning, training, validation, and exports. Stop after the final full validation, checkpoint and prediction files are written and hashed, with device synchronization. No free submission-specific caches, weights, or compiled graphs from a previous run. Imports, environment installation, downloads/checksum verification by `fetch_data.py`, CUDA context initialization, and final report-JSON writing are excluded. The runner rechecks data inside its clock. Shared inputs consist only of the released files.

Evaluation cadence may change but must be fixed across seeds and every check counts. Only actually evaluated models can qualify. Rank by the median of the three **inclusive** times. A failure invalidates the recipe's record attempt; do not omit it. A claimed improvement requires maintainer reproduction of all three seeds and a same-machine incumbent rerun, with a lower median and wins on at least two paired seeds. This small-panel rule is not a p < 0.01 significance claim.

## Rare-family audit

The remaining **31,148 molecules in 23,212 families** are a fixed, supplementary extrapolation audit. These families were absent from the pilot; the largest has 18 molecules. They are intentionally not described as a representative or distribution-matched test set. Fold assignment and label QC were frozen before this audit was evaluated.

After freezing the code and all three validation runs, evaluate the checkpoint from the first prescribed seed once. Report all five metrics, overall raw-f error, and the audit prediction file. The audit has no pass threshold and does not determine speed ranking; it shows where validation improvements fail to transfer. Do not tune on it. Labels are public in the source dataset and release, so this is a conventional holdout, not a secret test service. A changed recipe after audit exposure requires disclosure. Broad claims of chemical generalization are out of scope for v0.

## Submissions and verification

Follow [SUBMIT.md](SUBMIT.md). The automatic checker verifies the fixed data/scoring contract, seed panel, prediction files, reported metrics, and timing consistency. It cannot prove honest clocks, resource usage, data isolation, or absence of pretraining. Maintainers inspect and reproduce code before accepting a record; untrusted PR code is never automatically run on the cluster.

The thresholds were chosen after exploratory baseline calibration and frozen before the three release reference runs. Prior calibration timings are not records. Changes to data, target thresholds, or scoring create a new protocol version; v0 history remains comparable.
