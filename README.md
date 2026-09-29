# Mol Speedrun

**Train a molecular property predictor to fixed error targets as fast as possible.** One L40S, three seeds, chemically separated QCDGE molecules. Inspired by [nanoGPT](https://github.com/karpathy/nanoGPT) and [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt).

[Dashboard](https://mcox3406.github.io/mol-speedrun/) · [Rules](RULES.md) · [Data](DATA.md) · [Submit](SUBMIT.md)

```sh
pip install -r requirements.txt
python fetch_data.py                       # 11 MB download
python race.py --seed 20260929 --output runs/reference-20260929.json
```

Repeat with seeds `20260930` and `20260931`. Every seed must reach **MAE ≤ 0.180 / 0.150 / 0.115 eV** for S1 / T1 / gap, **≤ 0.250** for transformed absorption strength, and **≤ 0.120** on strong transitions. Rank by median elapsed time; 60-minute limit per seed.

Any architecture is welcome. Train from scratch on the supplied training set; count preprocessing, compilation, training, and evaluation. Submit code, all three runs, and predictions in a pull request. Records require maintainer reproduction.
