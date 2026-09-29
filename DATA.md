# Data and scope

Download with `python fetch_data.py` (about 11 MB compressed, 38 MB unpacked). The downloader checks the archive SHA256, fixed file list, and each file's checksum against `download.json` and `data-manifest.json`. It never extracts arbitrary archive paths. To use a local copy: `python fetch_data.py --archive /path/qcdge-v0.tar.gz`.

| File | Role |
|---|---|
| `train.csv` | 308,402 labeled structures for fitting |
| `validation.csv` | 64,549 structures for the speedrun targets |
| `audit.csv` | 31,148 rare-family structures; evaluate only after submission freeze |
| `split.json` | Training/validation IDs and inner training/tuning membership |
| `manifest.json` | Exact file hashes and counts |

CSV columns are `id,smiles,S1_eV,T1_eV,delta_ST_eV,f_S1`. SMILES are canonical isomeric strings. Inputs may include any features computed from these 2D structures, but not source-dataset geometry, calculated energies, other quantum descriptors, or IDs. S1/T1 are the lowest recorded singlet/triplet excitation energies; f is the oscillator strength of that S1 transition. These are vertical absorption calculations, not fluorescence measurements. Tied lowest singlets use the smallest oscillator strength, matching the frozen extraction code.

## Chemical separation and QC

For cyclic molecules, group by typed nonchiral Murcko scaffold; for acyclic molecules, use full generic topology. Union these groups with Standard InChI connectivity/mobile-H families. Preserve all pilot family assignments when expanding the cohort; families absent from the pilot form the audit. Exact canonical SMILES, group IDs, scaffold keys, and connectivity keys are disjoint across folds. This does not eliminate every close analogue or every tautomer relationship.

The inner training/tuning split has 261,923 / 46,479 molecules. Training/validation/audit contain 10,734 / 2,684 / 23,212 chemical groups. The audit is skewed toward rare families and measures that specific extrapolation. It is not a random molecule split.

The structural domain is single-component, net-neutral, carbon-containing, RDKit-closed-shell organic molecules with supported elements. Training heavy-atom 5th/50th/95th percentiles are 8/10/10. Of 404,188 unique structurally eligible molecules, exclude 54 repeated-canonical representatives (all known duplicate pairs quarantined), six malformed/unavailable excitation records, and 29 records with nonpositive excitation energy or negative oscillator strength. No small positive energy, gap, or intensity cutoff is used.

## Provenance and license

Source: **Quantum Chemistry Dataset with Ground- and Excited-state Properties of 450 Kilo Molecules**, [Scientific Data (2024)](https://www.nature.com/articles/s41597-024-03788-x), [QCDGE collection](https://doi.org/10.6084/m9.figshare.c.7259125.v1). The source [metadata](https://doi.org/10.6084/m9.figshare.25930258.v1) and [HDF5 archive](https://doi.org/10.6084/m9.figshare.25929007.v1) are released under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/). The derived data retain CC0; please cite the dataset paper and this benchmark.

We use **441,404 metadata-indexed records**, not every HDF5 root entry. The paper reports 443,106 molecules and the archive has 493,167 root entries; this discrepancy remains unresolved. v0 fixes the precise metadata-indexed cohort rather than claiming those counts agree. The archive MD5 is `605935abe6bc17efe841dd5b1253b21a`; metadata MD5 is `3c14fa79c7b869ded6eae3c654b873f7`.

Preparation and exclusion audits are in [experiments/full_cohort](experiments/full_cohort/README.md); earlier selection studies remain under `experiments/`. `package_data.py` packages the audited cohort without changing assignments. Large source data are unnecessary for ordinary competition runs.
