# Mol Speedrun

A deliberately small molecular property prediction speedrun **pilot**. Inspired by [nanoGPT](https://github.com/karpathy/nanoGPT) and the [modded-nanogpt rules](https://github.com/KellerJordan/modded-nanogpt#rules). No official target or hardware champion yet.

## Run it

Python 3.12; CPU baseline (no CUDA required).

```sh
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Frozen train/val/test CSVs are already included; to reproduce them:
python prepare.py
python train.py --model ridge --seed 0 --output submissions/ridge-0.json
python train.py --model plain --seed 0 --epochs 20 --output submissions/plain-0.json
python train.py --model string --seed 0 --epochs 20 --output submissions/string-0.json
python train.py --model graph --seed 0 --epochs 20 --output submissions/graph-0.json
python -m unittest discover -s tests
python dashboard.py
python -m http.server 8765 --directory docs --bind 127.0.0.1
```

Open http://127.0.0.1:8765. Dashboard is a generated static file, with a run selector, learning curves, and a results table. No server database. Rebuild after adding JSON. A public release can serve `docs/` through GitHub Pages; this initial repository is private.

## What is implemented

- `mean`: training-label mean; a sanity floor.
- `ridge`: radius-2, 2,048-bit Morgan fingerprints + ridge regression, alpha=10.
- `plain`: ASCII character tokens, learned positions, two 64-wide bidirectional transformer layers, masked mean pooling, scalar regression.
- `string`: same encoder with a 2,048-row hashed bigram embedding and learned elementwise gate at its input. A small inspired ablation, **not a reproduction of MolGram/Engram**.
- `graph`: same encoder with a gated projection of Morgan bits at the pooled readout. This is learned pooled fingerprint fusion, **not atom-aligned graph memory**. Parameter budgets and injection locations differ from `string`; these runs cannot isolate representation alone.

Only training targets determine normalization. Training uses normalized MSE; reports use RMSE in original log10 mol/L units. Fixed ASCII vocabulary avoids fitting a tokenizer on held-out structures. Full molecules, no truncation; reject strings over 512 characters. No pretraining, augmentation or external descriptors. Fingerprints are computed from the provided structure.

## Data contract: esol-scaffold-v0

[ESOL via DeepChem](https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv) contains 1,128 rows. Predict **measured** log solubility; the supplied ESOL prediction and descriptor columns are excluded. Attribution: Delaney, *ESOL: Estimating Aqueous Solubility Directly from Molecular Structure*, J. Chem. Inf. Comput. Sci. 44 (2004), 1000–1005, https://doi.org/10.1021/ci034243x; [MoleculeNet](https://doi.org/10.1039/C7SC02664A).

`data/manifest.json` pins raw SHA-256, RDKit version, processed file hashes, counts and policy. Canonical isomeric SMILES, no salt stripping; duplicate measurements retained in the same scaffold group. Sort Bemis–Murcko scaffold groups by decreasing size, lexical tie break; fill train to at most 902 and validation to at most 113, otherwise test. All acyclic molecules share the empty scaffold and stay together. This is our custom split, not a claim to reproduce a published MoleculeNet score. Scaffold separation does not eliminate all chemical similarity.

The test labels are public but unused by the trainer. Treat them as a lockbox by convention; this is not a secret evaluation service. Do not tune on them. At protocol freeze, evaluate prespecified finalists once using a separate audited evaluation script (not implemented in this pilot). Repeated public validation tuning can still overfit.

## Pilot submission rules

1. Preserve data, labels, split membership, target units, and evaluation formula. Submit new featurization as code. No external data or pretrained weights in this track.
2. Include exact command, code commit, software versions, hardware model, CPU threads, parameter count, and every run JSON. Record unsuccessful runs too. Generated runs include source/manifest hashes and a dirty-tree flag. Do not edit measurements.
3. Use seeds 0–4 for comparisons, a fixed epoch budget, and report all learning curves. The checked-in one-seed, short smoke runs only demonstrate operation. Ridge and mean are deterministic and do not acquire uncertainty by rerunning seeds.
4. The timer begins before CSV loading/featurization, and includes model construction, training and every full validation. Imports/environment setup/download and JSON serialization are excluded. No free model-specific caches. CPU only in v0; a GPU track needs synchronized timing and peak-memory measurement before records are accepted.
5. Compare speed only on identical hardware/thread/software settings; rerun baseline and candidate on the same machine. No cross-device speed ranking. Record preprocessing time separately as well as inclusive elapsed time.
6. The dashboard displays exploratory runs, not accepted records. JSON validation cannot prove correctness; maintainers must inspect code and reproduce results. PRs are the submission mechanism. Include all seeds and update `docs/index.html` via `python dashboard.py`.

## Before an actual speedrun

Calibrate a useful RMSE threshold with repeated baselines, then **freeze it**, the training budget, validation cadence, reference hardware and protocol version before accepting challenges. Record first threshold crossing; a failure to reach it is a failure, not an omitted run. Choose a fixed seed panel and significance procedure in advance (a one-sided 99% upper confidence bound on mean RMSE below the target is a starting point). Avoid optional stopping over seeds/checkpoints. Training-seed statistics describe this split, not broad chemical generalization. A separate held-out audit and later split replication are necessary.

ESOL makes setup cheap but is too small for a robust systems speed race. Next candidate: a larger single-task regression dataset, e.g. a carefully selected QM9 target if quantum properties are the question. Revisit task relevance, units, stereochemistry/geometry, splits and labels before promoting anything to v1. Experimental solubility and quantum energies answer different questions.

The first scientific comparison should add parameter-matched controls and atom-aligned graph environments. Hashed string memory versus pooled fingerprints is only a starting ablation. Check canonical versus randomized SMILES later, with augmentation restricted to training and its cost counted.

## Layout

`prepare.py` → frozen `data/`; `train.py` → `submissions/*.json`; `dashboard.py` → `docs/index.html`. CI tests split integrity, padding invariance and malformed result rejection, validates JSONs and checks that the dashboard is regenerated. It does not run submitted training code as a trusted record or grant records automatically.

## Research motivation

[MolGram (June 2026)](https://arxiv.org/abs/2606.12113) studies generation, forward reaction prediction and retrosynthesis. It motivates testing local memory here; it does not establish an advantage on property prediction. [RDKit Morgan fingerprints](https://www.rdkit.org/docs/GettingStartedInPython.html#morgan-fingerprints-circular-fingerprints) provide an inexpensive graph-derived control. A collision is information loss; skewed motif frequencies do not guarantee that collisions are harmless.
