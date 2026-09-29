# Harder task screen: excited states

**Decision:** prioritize excited-state regression; do not freeze a dataset or quality threshold yet. QCDGE is the leading next dataset to investigate. Tartarus demonstrates useful difficulty, but its label disagreement and topology concentration need resolving. New quantum calculations are not justified at this stage.

## Measured evidence

Downloaded Tartarus `gdb13.csv`: 398,453 rows, not the README's nominal 403,947. Full audit: 386,005 unique canonical isomeric structures, 90,388 typed scaffolds, but only **64 generic ring/linker cores** after removing terminal scaffold leaves. There are 11,315 repeated-structure groups; 1,373 have gap ranges above 0.1 eV. Median/p95 repeated-label ranges are 0.00018/0.242 eV for the gap and 0.0000085/0.0844 for oscillator strength. These are observed disagreements, not an estimated irreducible-error bound. Different geometry, convergence, or source processing could explain them; provenance must be checked.

A seeded 100k-row development sample retains 99,191 unique structures. No random molecule split: entire scaffold groups are separated in both outer evaluation and inner model selection. First duplicate retained, never averaged across partitions. This pilot does not establish a final duplicate policy or tautomer-family separation.

| Holdout / model | Gap MAE (eV) | Gap R² | Transformed oscillator MAE |
|---|---:|---:|---:|
| Typed scaffold / median | 0.3220 | −0.004 | 0.7756 |
| Typed scaffold / Morgan ridge | 0.2452 | 0.395 | 0.6060 |
| Typed scaffold / descriptor boosting | 0.2519 | 0.363 | 0.5918 |
| Typed scaffold / fingerprint MLP | **0.1797** | **0.646** | **0.4500** |
| Generic core / median | 0.3064 | −0.001 | 0.7429 |
| Generic core / Morgan ridge | 0.2582 | 0.255 | 0.6179 |
| Generic core / descriptor boosting | 0.2818 | 0.089 | 0.6282 |
| Generic core / fingerprint MLP | **0.2301** | **0.390** | **0.5158** |

Oscillator transform is `log10(1 + f/0.001)`, an exploratory floor that avoids optimizing numerical noise in nearly dark transitions. Raw-f errors and brightness strata are also recorded. On the generic split, MLP raw-f MAE is 0.0536 versus 0.0670 for always-dark; on f≥0.1 molecules those errors are 0.1322 versus 0.2052. Do not declare success from pooled raw-f MAE alone.

Typed train/validation counts: 78,814/20,377. Generic: 71,676/27,515, only 50/13 cores; one core contains 17,747 validation molecules. Generic nearest-train Morgan similarity median is 0.487 across 300 queries; none reach 0.8. This coarse split is a stress test, not a suitable sole leaderboard split. The two splits have different evaluation molecules; metric differences do not isolate splitting as a causal effect.

The MLP is a joint two-output 256/128 network, AdamW, batch512, normalized L1 loss. Hyperparameters/epochs use training-only group holdouts; final refits use a fixed selected epoch count. One seed, CPU, cached features: **not speedrun timings or proof that all cheap methods fail**. Stronger fingerprints, kernels, trees and full-data learning curves remain necessary. No hidden test was evaluated.

## Proposed speedrun

Predict S1 and T1 excitation energies, with an explicit S1−T1 error gate; consider oscillator strength only after its label audit. Requiring each target's tolerance avoids one easy target compensating for another. A scalar `max(MAE_j / tolerance_j)` gives a time-to-quality finish line at ≤1. Fix tolerances using baseline and GPU calibration, before accepting submissions.

Use one specified high-end GPU, aiming for an initial competitive run of 10–60 minutes. Inputs are molecular structure only. Group duplicates, stereoisomers and tautomer families without changing label-bearing structures. Build train/development/test with structural-similarity-aware grouping, audit nearest neighbors and group concentration, and reserve an untouched final test. Typed scaffolds alone are insufficient; extremely coarse generic scaffolds can also be unsuitable. Define the chemical transfer setting first; do not optimize splits against baseline errors. Before freezing, compare several development partitions and use group-level uncertainty.

All architectures, including cheap ML, remain eligible. Count model-specific preprocessing, compilation, initialization and training; report final evaluation separately. This differs deliberately from the warmup-excluding clock in the [39.9-second nanoGPT writeup](https://hyperstition.cc/training-nanogpt-in-39-9-seconds). Frozen data, reproducible commands, all-seed reports and PR-based submissions transfer directly. Do not promise a threshold until a transformer/GNN demonstrably reaches it on the target GPU.

## Existing-data shortlist

- **QCDGE:** 443,106 molecules, ten singlet and ten triplet states, up to ten C/N/O/F heavy atoms. Excited-state labels use ωB97X-D/6-31G(d). Broader starting point than the narrowly filtered emitter set; not yet baseline-tested here. [Paper](https://www.nature.com/articles/s41597-024-03788-x), [data collection](https://doi.org/10.6084/m9.figshare.c.7259125.v1).
- **Tartarus:** current empirical stress test. It targets conjugated cyclic emitters; the source workflow uses B3LYP/6-31G* excited-state calculations following conformer search. We predict the CSV's positive gap and oscillator columns, not the composite design score, calculation time, or the API's negated gap reward. This is a computational surrogate task, not experimental emission prediction. [Source](https://github.com/the-matter-lab/Tartarus), [methods supplement](https://papers.nips.cc/paper_files/paper/2023/file/09f8b2469a3d1089a7c60d9ef1983271-Supplemental-Datasets_and_Benchmarks.pdf).
- **ORNL_AISD-Ex:** about 10.5M molecules, TD-DFTB spectra. Larger, lower-fidelity alternative with much heavier data handling; size alone does not make it preferable. [Official record](https://doi.ccs.ornl.gov/dataset/13423cfb-df80-541c-a3d9-a2f042fbe507).

QCDGE's metadata CSV has no targets. The full HDF5 is 48.46GB; individual source subsets are 10–13GB. This laptop does not have room for the full archive. A storage-capable machine can extract a compact label table from the existing archive—**no quantum calculation required**. The published README identifies `excited_state/Info_of_AllExcitedStates`; its internal schema/units must be inspected before building a parser. No untested extractor is presented as working.

Similarity-aware splitting references: [DataSAIL](https://doi.org/10.1038/s41467-025-58606-8), [Lo–Hi](https://arxiv.org/abs/2310.06399). Their drug-discovery findings motivate leakage audits; they do not validate a particular quantum-property split for us.

## Reproduce

From `experiments/task_selection`, using the repository environment:

```sh
curl -L --fail https://raw.githubusercontent.com/aspuru-guzik-group/Tartarus/main/datasets/gdb13.csv -o raw/tartarus-gdb13.csv
python emitter_audit.py --csv raw/tartarus-gdb13.csv --out results/emitter-audit.json
python emitter_screen.py --csv raw/tartarus-gdb13.csv --work results --split typed
python emitter_screen.py --csv raw/tartarus-gdb13.csv --work results --split generic
```

Expected source SHA256: `c3d4dec043010ffa069c38acd2059d4685a44d9479f04a9e7a7601b0a0b567b2`. Result JSON includes source indices, versions, inner tuning scores, script digest and evaluation diagnostics. Verify the source hash when reproducing because the upstream main-branch URL can change. Caches assume an unchanged source and featurizer; delete them when either changes.
