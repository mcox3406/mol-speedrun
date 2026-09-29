# QCDGE: energy and transition-intensity pilot

**Recommendation:** develop a joint excitation-energy and transition-intensity speedrun. Energy-only prediction is already quite strong with cheap models. Intensity adds useful difficulty, but full-data learning curves and a GPU transformer/GNN run must establish the attainable finish line before freezing rules.

## Actual results

A fixed 100k-ID sample spans all four source collections. After structure checks and physical-validity checks, there are **75,280 training / 16,257 development-validation molecules**. Entire chemical groups are separated in outer evaluation and inner tuning; this is not a random molecule split.

| Model | S1 MAE, eV | T1 MAE, eV | Gap MAE, eV | Log-intensity MAE |
|---|---:|---:|---:|---:|
| Morgan ridge | 0.507 | 0.640 | 0.394 | 0.438 |
| Descriptor boosting | 0.435 | 0.403 | 0.283 | 0.419 |
| Morgan + descriptor forest | 0.346 | 0.314 | 0.227 | 0.350 |
| Morgan + descriptor MLP, three-seed mean | **0.269** | **0.240** | **0.171** | **0.325** |

MLP energy MAE standard deviations across three training seeds: 0.0018/0.0037/0.0032 eV. These do not measure uncertainty across chemical splits. Energy predictions determine the gap by subtraction. Intensity uses a separately trained scalar model, so the table is not a measured joint-model speedrun.

Intensity is `log10(1 + f_S1/0.001)`. The floor is exploratory. MLP mean R² is 0.642 on this transform and 0.326 on raw intensity. Raw-f MAE is 0.0353 versus 0.0502 for always-dark; on the 2,202 validation molecules with f≥0.1, it is **0.188 versus 0.309**. A pooled raw-f score alone would substantially underweight failures on bright transitions. These are vertical excitation/absorption properties of computed structures, not experimental fluorescence labels.

Models use only Morgan bits and/or 16 inexpensive structural descriptors, never other quantum properties, source IDs, coordinates, or runtimes. Ridge/boosting/forest choices and MLP epoch counts are selected inside training. MLP architecture is 256/128 hidden units, AdamW, batch512; energy loss averages normalized S1/T1/gap errors. The energy model selects 10 epochs; intensity selects six. CPU fits exclude cached feature construction and some overlap in execution: **no eligible speed records are claimed**.

## Split and data audit

- Full metadata structural grouping uses typed, nonchiral Bemis–Murcko scaffolds, generic whole-molecule topology for acyclic molecules, and union with Standard InChI first-block families. This links stereoisomers and InChI-recognized mobile-H tautomers without rewriting label-bearing structures. It is not proof of separation for every possible tautomerization.
- Domain: single-component, net-neutral, carbon-containing structures with zero RDKit-assigned radical electrons. This operational rule does not prove a singlet ground state. Structure checks retain 404,188 unique molecules in 36,645 groups before sampling labels.
- Final train/validation contain 10,734/2,684 groups. Largest validation group: 1,135 molecules (7.0%). Nearest-training Morgan similarity across 500 validation queries: median **0.471**, with 1.6% ≥0.8. Acyclic fractions differ: 22.4%/28.0%. Similarity and distribution audits remain necessary beyond a scaffold label.
- Verified files contain **441,404 metadata rows** and **493,167 HDF5 root entries**, versus 443,106 molecules in the paper. This discrepancy is unresolved. Extraction samples IDs explicitly present in the publisher's metadata; it does not treat every archive entry as curated data.
- Two sampled records, Aa23/Aa24, contain an unrecognized spin label (`1.000`) and fail strict parsing. Four otherwise eligible sampled records have nonpositive excitation energies; one also has negative f. The final screen requires S1>0, T1>0, f≥0. No accuracy-driven gap, size, or intensity cutoff is used. Excluded IDs and values are recorded.
- Across the full eligible structure pool, there are 54 canonical duplicate pairs. 31 pairs differ by >0.1 eV in at least one energy target; maximum S1 disagreement is 2.32 eV. The pilot keeps the first metadata representative. Before release, quarantine repeated structures or establish a documented conformer/replicate policy. Disagreement is not an estimated noise floor for all molecules.
- A separate 4,000-molecule audit found no S1 energy ties at published precision. The current parser uses the smallest f among exactly tied lowest singlets; a frozen protocol should define the degenerate-state policy explicitly.

No independent final test has been used. Pre-QC results are retained for transparency, but removing groups changed partition assignment; they are not a controlled estimate of the cleaning effect.

## Next gate before a speedrun launch

Fit full-data strong cheap controls and a small SMILES transformer/message-passing model. Freeze a structure-only chemical transfer split, reserve an untouched test, and calibrate per-target tolerances on the intended GPU. Require energy, intensity, and bright-transition criteria simultaneously. Do not choose the split by maximizing baseline error. New quantum calculations remain unnecessary.

## Files and reproduction

The external drive contains the verified archive, extracted labels, features and `development/train.csv`, `development/validation.csv`, and their checksum manifest at `/Volumes/sandisk2tb/mol-speedrun/qcdge/`. The exports are a development pilot, not a frozen competition release. Source manifests, chemical split IDs, label audits and scores are committed under `results/qcdge-*`.

From `experiments/task_selection`, with the pinned repository environment:

```sh
python extract_qcdge.py --directory /Volumes/sandisk2tb/mol-speedrun/qcdge --sample 100000 --workers 4
python qcdge_features.py --directory /Volumes/sandisk2tb/mol-speedrun/qcdge
python qcdge_screen.py --directory /Volumes/sandisk2tb/mol-speedrun/qcdge --out results
python qcdge_oscillator.py --directory /Volumes/sandisk2tb/mol-speedrun/qcdge --out results
python qcdge_duplicate_audit.py --directory /Volumes/sandisk2tb/mol-speedrun/qcdge --out results/qcdge-duplicates.json
python qcdge_intensity_audit.py --directory /Volumes/sandisk2tb/mol-speedrun/qcdge --results results
python export_qcdge_split.py --directory /Volumes/sandisk2tb/mol-speedrun/qcdge --manifest results/qcdge-100000-indices.json
python -m unittest discover -s . -p 'test_*.py'
```

Extraction is resumable. `--sample 0` requests all metadata IDs; it has **not** been run to completion. The cached label CSV also contains an initial sequential batch, which the baseline excludes unless its IDs occur in the fixed sample. Never run concurrent writers to that CSV. Intermediate caches assume unchanged verified sources.

[QCDGE paper](https://www.nature.com/articles/s41597-024-03788-x) · [Publisher's data collection](https://doi.org/10.6084/m9.figshare.c.7259125.v1) · [Earlier candidate screen](EXCITED_STATES.md)
