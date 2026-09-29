# Submit a run

1. Read [RULES.md](RULES.md), download the data, and freeze your code, hyperparameters, and evaluation/stopping schedule.
2. Run all three prescribed seeds in separate processes on the reference allocation. Keep every attempt, including failures. The reference command is `python race.py --seed 20260929 --output runs/reference-20260929.json`; repeat for `20260930` and `20260931`.
3. Once all three runs are frozen, audit the first seed's checkpoint. For the reference model:

   ```sh
   python audit.py --run runs/reference-20260929.json --output runs/audit.json
   ```

   Other architectures provide their own checkpoint loader, the same audit predictions format, and equivalent checkpoint/run hashes. Do not tune on this result.
4. Recompute all metrics with the independent checker:

   ```sh
   python validate_submission.py \
     --runs runs/reference-20260929.json runs/reference-20260930.json runs/reference-20260931.json \
     --audit runs/audit.json --output runs/submission.json
   ```

5. Open a PR with code, the exact command/environment, hardware allocation, development search summary, and all run JSONs/logs. Put lightweight reports under `records/<name>/`. Attach compressed prediction files and checkpoints as release assets or another persistent download; include their SHA256s. Keep large binaries out of Git. Include failures and disclose any earlier audit exposure.

Reports must retain the reference schema (see `race.py`): protocol/data hashes, seed, hardware, source commit/hashes, inclusive clock, evaluation history, metrics, prediction/checkpoint hashes, and software versions. The frozen evaluator and data cannot change. You may modify the model/trainer, preprocessing and evaluation cadence within the rules.

Passing `validate_submission.py` means **checks passed, reproduction pending**. It does not award a record or certify the clock. Maintainers review the code and rerun all three seeds, plus the incumbent on matching hardware/software, before adding an accepted entry to `records/index.json`. PR workflows check trusted scoring/tests; they do not execute submitted training code on the cluster. No account system or database is needed.

The supplied baseline is a reference measurement, not a statistically certified comparison against all architectures. Three prescribed seeds are a reproducibility panel, not a p-value.
