# Choosing the first molecular speedrun

**Use PCQM4Mv2 orbital-gap prediction. Establish the chemically separated task first, then calibrate the speed target on one H100.** These are development results, not a frozen benchmark or evidence that a neural model already reaches a useful target within an hour.

## What the local experiments say

The primary screen contains **156,627 training molecules and 36,122 chemically held-out validation molecules** after the domain filter. Whole ring scaffolds stay together; acyclic molecules are grouped by full untyped heavy-atom topology. Inner tuning uses the same chemical separation.

| Baseline | Validation MAE (eV) ↓ |
| --- | ---: |
| median | 0.920 |
| atom_counts_ridge | 0.573 |
| descriptors_ridge | 0.473 |
| morgan_ridge | 0.432 |
| descriptors_boosting | 0.334 |
| Fingerprints + small MLP | 0.242 |

The MLP is only 565,249 parameters and its final refit/evaluation takes about 16 CPU seconds with cached features. This excludes feature construction and search and is not a speedrun measurement. **A 0.25 eV target would already be too easy.** Neither these numbers nor the published OGB numbers justify freezing a harder target yet.

On the same *unfiltered* chemical holdout, increasing the MLP's training set from 50k to 174,227 molecules improves MAE from 0.313 to 0.268 eV. A full-scale fingerprint baseline may therefore improve further. Before launch, it must receive the same full training pool and a reasonable tuning budget as the transformer/GNN.

QM9 is a useful diagnostic but a smaller, narrower chemical universe: this sample's 5th/50th/95th heavy-atom percentiles are 8/9/9. PCQM4Mv2 spans 10/15/17 in the primary training sample and 9/14/17 in validation. A QM9 atom-count model gives almost perfect total-energy R² while missing by 0.888 eV MAE. Total-energy R² would be a misleading contest objective.

## Molecules and split

For the first track, I recommend **neutral, single-component, carbon-containing molecules, no RDKit-assigned radical electrons**, with elements H/B/C/N/O/F/Si/P/S/Cl/Br/I. This is an operational 2D domain definition, not independent confirmation of the DFT spin state. Do not filter by target magnitude, conjugation, prediction error, or arbitrary molecular-weight thresholds.

The raw sampled pool had 7.85% radical-bearing structures; domain filtering retains 192,749 of 209,369 unique parseable structures. The split keeps cyclic fractions close (88.5% train, 87.3% validation), with mean gaps 5.754 and 5.783 eV. In a 500-molecule validation audit against **all** training molecules, median nearest-neighbor Morgan similarity is 0.50 and 1.4% reach 0.8. Scaffold separation is not a guarantee of low similarity; retain this audit, including per-similarity-bin errors, when scaling up.

Two tempting shortcuts failed the audit: ordinary empty-scaffold grouping made validation entirely cyclic; fully generic ring scaffolds plus acyclic topology shifted validation to 31% acyclic versus 10% in training. Typed ring scaffolds with the acyclic fallback gave a more representative comparison here. This choice concerns molecular coverage, not choosing the split with the worst model score.

The full release should assign these chemical groups once to train/validation/final-test **before** drawing nested training subsets. Freeze the IDs and report group sizes, similarity distributions and label coverage. The official OGB split is by CID, so our numbers must never be compared directly to its leaderboard. The screening pool used public labels only; its validation has now informed task design and is not an untouched final test.

Duplicate audit: 630 duplicate pairs, median gap difference 0.026 eV, 95th percentile 0.448 eV, four above 1 eV. The pilot keeps the first source row. Before freezing full data, investigate this disagreement and lock a structure-level replicate policy; do not silently treat it as model error or put conformer/tautomer/stereoisomer families across splits. Preserve structures as supplied rather than neutralizing/tautomerizing them and assuming their DFT labels remain valid. Current grouping handles stereo variants with the same scaffold but does **not** establish full tautomer-family separation; that audit remains open.

## Proposed race

- **Task:** scalar HOMO–LUMO gap from 2D structure, MAE in eV. Keep the full eligible PCQM4Mv2 training pool available; let submissions choose sampling schedules. No external data, pretrained models or supplied 3D geometry in this initial track.
- **Hardware:** one precisely specified H100 80 GB configuration, fixed host resources and environment. Aim for an initial reference run of 10–60 minutes. This has not been timed on a GPU yet.
- **Threshold:** choose only after a compact transformer/GNN and strong fingerprint/boosting/kernel controls run on that exact full-data split. Investigate 0.15 eV as a calibration point, **not an announced threshold or claim of attainability**. A qualifying target should be convincingly beyond the cheap controls and reliably reachable by the reference within the time budget. If those conditions fail, revise the task before launch.
- **Clock:** I propose counting model-specific preprocessing, compilation, initialization, training and final averaging; exclude download/install and report final evaluation separately. This makes moving work into an untimed cache unhelpful. Publish stage timings and examples/updates processed so faster steps and fewer needed updates can be distinguished.
- **Qualification:** freeze the schedule and seed panel before certification; five complete independent runs, same hardware, all outcomes logged. Prespecify the statistical quality gate after calibration. Permit any model, including cheap ones; architecture restrictions should not manufacture difficulty. Audit finalists once on the reserved chemical test split.

[Hyperstition's writeup](https://hyperstition.cc/training-nanogpt-in-39-9-seconds) motivates controlled ablations and reporting both convergence and step cost. Its primary clock excludes compilation/warmup, unlike the proposed inclusive clock here. Optimizers, averaging and sparse memory are relevant experiments; sampled vocabulary softmax is not relevant to a scalar regression head.

The blog should wait for the threshold/GPU calibration decision, then become a short introduction to this one task. The paper survey and infrastructure details belong in repository documentation.

## Sources

- [PCQM4Mv2 task, provenance, official split and CC BY 4.0 license](https://ogb.stanford.edu/docs/lsc/pcqm4mv2/).
- [OGB reference baselines](https://github.com/snap-stanford/ogb/tree/master/examples/lsc/pcqm4m-v2): evidence of learnability on their split, not a calibration result for ours.
- [DeepChem QM9 source, units and sanitization caveats](https://github.com/deepchem/deepchem/blob/master/deepchem/molnet/load_function/qm9_datasets.py).

Exact commands, source digests, split indices and limitations are in [README.md](README.md) and `results/`.
