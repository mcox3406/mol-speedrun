# Initial smoke runs

These five runs used the same Apple M2 Pro CPU, Python 3.12.8 and the pinned environment in `requirements.txt`, with four Torch CPU threads. They ran sequentially on September 29, 2026. All used seed 0 and the clean training revision recorded in each JSON. These are short functionality checks, not tuned results or accepted records. No test evaluation was performed.

Exact commands (from repository root):

```sh
python train.py --model mean --output submissions/mean-0.json
python train.py --model ridge --output submissions/ridge-0.json
python train.py --model plain --epochs 3 --output submissions/plain-0.json
python train.py --model string --epochs 3 --output submissions/string-0.json
python train.py --model graph --epochs 3 --output submissions/graph-0.json
```

The `epochs` argument is unused for mean/ridge; each has one recorded evaluation. OS scheduling and first-use overhead can affect these very short CPU times. The observed ordering is not evidence of a statistically reproducible advantage. Do not compare these three-epoch runs against longer runs as if they had equal training budgets.

For a new submission, use a unique filename and include exact commands, hardware model, dependency versions, fixed budget, every prespecified seed, code revision, changes from the baseline, and a discussion of failures. Rebuild the dashboard. Tests validate format and data integrity; they cannot authenticate reported measurements.
