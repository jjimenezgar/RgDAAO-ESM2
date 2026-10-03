# Follow-up robustness protocol

**Status: implemented and tested; full GPU training is pending.**
The completed [initial benchmark](results/seed42/README.md) remains unchanged.
This follow-up was designed after inspecting that result. It keeps the same
model, representation, learning rates, epoch budget and validation-selection
metric; it does not tune them against the new test results.

## Two questions

1. Does the improvement persist across training seeds on the same random split?
2. Does it persist when test mutation positions are absent from training?

The default plan contains both `random` and `position` strategies, paired
frozen/fine-tuned models, and training seeds **42, 43, 44**: twelve model runs.
Each strategy uses one fixed partition generated with **split seed 42**.
Model initialization, dropout and training-loader order vary across training
seeds; data partitions do not. Both models share the same training seed in
each pair. The manifest explicitly authorizes the allowed training seeds.

## Position-held-out partition

All single substitutions at one residue belong to the same partition.
Positions are sorted numerically, permuted with NumPy `default_rng(42)`, and
allocated 80/10/10 by **position count**, with integer truncation for train
and validation. Row counts need not be exactly 80/10/10. Group allocation uses
no activity labels and makes no adjustments to balance test difficulty.

| Partition | Positions | Variants |
|---|---:|---:|
| Train | 291 | 5,119 |
| Validation | 36 | 638 |
| Test | 37 | 642 |

These counts were verified against the actual 6,399-variant dataset.
[Prepared position manifest](audit/position_split_summary.json) records the
position identities, label distributions and CSV hashes. These are data audits,
not model performance. Training checks position disjointness for train/validation;
final evaluation checks all three partitions. Exact mutation/sequence overlap
checks continue to apply to both split strategies.

The random strategy reproduces the original **5,119 / 639 / 641** split CSVs
byte-for-byte. Its seed-42 pair is rerun in the follow-up rather than mixing
the original archived run with newly executed runs.

## Run in Colab

[Open robustness notebook](https://colab.research.google.com/github/jjimenezgar/RgDAAO-ESM2/blob/main/notebooks/RgDAAO_ESM2_robustness.ipynb).
Use a fresh GPU runtime. The notebook defaults to `STRATEGIES = ['position']`
to prioritize the position-held-out experiment. Run `['random']` in a second
fresh session, or select `['position', 'random']` for the full suite.

Each strategy runs six models. Based on the initial T4 runtime, budget roughly
three hours per strategy plus setup/export; this is an estimate, not a measured
runtime for the new suite. Download the evidence before the runtime expires.
The notebook keeps the original benchmark notebook intact.

## Run locally

After installing the ML extras, downloading the deposit and building
`data/processed/variants.csv` as described in the main README:

```bash
# Lock and inspect the plan without loading ESM-2 or using a GPU.
python scripts/run_robustness.py --prepare-only --output results/robustness

# Execute the saved plan. All training/selection finishes before test evaluation.
python scripts/run_robustness.py --resume --output results/robustness

# Recompute summaries from saved predictions; no new model evaluation.
python scripts/run_robustness.py --summarize-only --output results/robustness
```

For one strategy, pass `--strategies position` or `--strategies random` at
preparation. The original single-run workflow still works with its existing
configs and default preparation arguments.

New output directories are required for new plans. `--resume` takes its settings
from the saved `plan.json`, checks configuration/data hashes, and skips completed
training and evaluation stages. It refuses to train additional models once
test evaluation has begun. It does not resume an interrupted optimizer run:
preserve partial evidence and start a new complete plan in a fresh directory.
The Colab export excludes intermediate optimizer checkpoints while retaining
selected models and full run evidence.

## Outputs and interpretation

The result directory contains the locked plan, complete split CSVs, provenance,
reference sequence, per-run configuration/environment/history/checkpoint/test
evidence, prediction figures and validation curves. Summaries are written only
after every planned run has completed final evaluation:

| File | Contents |
|---|---|
| `runs.csv` | Four test metrics for each split/seed/model |
| `aggregate.csv` | Model metric mean and sample SD across training seeds |
| `paired_deltas.csv` | Fine-tuning improvement for each split/seed/metric |
| `paired_summary.csv` | Mean and sample SD of paired improvements |
| `summary.md` | Readable model comparison table |

Positive paired improvement means tuned-minus-frozen for correlations and
frozen-minus-tuned for errors. Sample SD uses `ddof=1`.
The summary verifies recorded metrics against predictions, checks planned test
identifiers and labels, and rejects changed configurations, splits or artifacts.

Three seeds estimate training variability on these fixed partitions. They do
not provide a confidence interval over possible splits or test observations,
and no statistical significance claim is made. Position-held-out performance
still concerns single substitutions in the same enzyme and assay; it does not
establish generalization to different proteins or multi-mutants. Report both
strategies and every planned seed, including negative or mixed results.
