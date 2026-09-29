# Dashboard

`dashboard.py` fills `index.html` with the frozen protocol and recorded measurements.
The generated page and its CSS/JavaScript are served from `docs/`.

The molecule explorer uses four **training-only** examples. `docs/molecules.json`
contains their frozen labels, 2D drawings, and illustrative ETKDG/MMFF conformers;
these geometries are not QCDGE geometries or model inputs. Regenerate with the
research environment (RDKit 2026.03.6):

```sh
python scripts/dashboard_molecules.py --train /path/to/qcdge-v0/train.csv
python dashboard.py
```

The energy-level diagram uses the actual labels and a shared linear eV scale.
The oscillator strength is reported separately; no orbital or spectral data are
implied. The 2D representation remains available when WebGL is unavailable.
