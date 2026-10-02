# Benchmark protocol

## Question and data

Does supervised fine-tuning of `facebook/esm2_t6_8M_UR50D` improve prediction
of direct EP-Seq activity fitness relative to a frozen backbone on RgDAAO
single substitutions? [Data provenance](../data/README.md) specifies the
exact source, numbering, filtering and reference sequence.

The input is the 365-residue mutated protein sequence. The target is the
unchanged experimental activity fitness. Expression is retained for potential
later interpretation; the model never consumes it.

## Fixed comparison

| Setting | Frozen | Fine-tuned |
|---|---|---|
| Pretrained model | ESM-2 t6 8M | Same checkpoint |
| Revision | `c731040fcd8d73dceaa04b0a8e6329b345b0f5df` | Same |
| Representation/head | HF ESM CLS token → dense/tanh/dropout → scalar | Same |
| Trainable parameters | 103,041 | 7,840,442 |
| Total parameters | 7,840,442 | 7,840,442 |
| Backbone | Frozen, eval mode | Trainable |
| Head learning rate | 1e-3 | 1e-3 |
| Backbone learning rate | No backbone updates | 1e-5 |
| Optimizer | AdamW, weight decay 0.01 | Same |
| Schedule | Linear decay, no warmup | Same |
| Epochs / batch size | 20 / 8 | Same |
| Random seed | 42 | Same |
| Loss | MSE on deposited labels | Same |
| Checkpoint selection | Maximum validation Spearman | Same |

The head is initialized identically through seeding before model construction.
Full precision, deterministic PyTorch algorithms, fixed loader seeds and no
sequence truncation are used. Max gradient norm is Trainer's default 1.0.
Both receive the same number of training examples and epoch opportunities.
No hyperparameter search or test-based tuning is performed. The smaller
backbone rate protects pretrained parameters while the common head rate permits
both newly initialized heads to learn on equal terms.

## Splits and test isolation

The existing seed (42), NumPy `default_rng` permutation and 80/10/10 allocation
are retained. Rows are lexicographically sorted by mutation before splitting.
Train and validation counts use integer truncation; the remainder goes to test:

| Split | Variants | Activity mean | Activity SD |
|---|---:|---:|---:|
| Train | 5,119 | −0.334951 | 0.275199 |
| Validation | 639 | −0.334514 | 0.274893 |
| Test | 641 | −0.349763 | 0.276398 |

These are label distributions, not model performance. Full statistics and
CSV SHA-256 values are in [split_summary.json](audit/split_summary.json).
They are recorded during data preparation and never used for fitting.

Training reads only train/validation CSVs. The runner completes **both**
training/selection stages before calling final evaluation. Validation Spearman
is calculated at every epoch; the best checkpoint is restored and saved.
Undefined validation correlation fails loudly. Evaluation verifies the saved
checkpoint and split hashes, validates cross-split disjointness, and writes
predictions in test-row order. Existing experiments and repeated test evaluations
are rejected to prevent accidental overwrites or unnoticed reruns.

## Metrics and artifacts

Primary metric: Spearman rank correlation. Also report Pearson, MAE and RMSE.
MAE/RMSE use the original deposited fitness units. Constant targets/predictions
have undefined correlations serialized as JSON `null`, not zero.

Each experiment saves configuration, seed, checkpoint revision, parameter
counts, software versions, git commit/dirty status, package freeze, split/data
manifests, epoch history, best validation metrics, and selected model/tokenizer.
Final evaluation adds `metrics.json`, `predictions.csv` and test metrics in
`run.json`. Plotting verifies completed evaluation and prediction hashes before
creating experimental-vs-predicted PNG/PDF panels and a comparison CSV.

## Status and limitations

The real pretrained CPU smoke test passes in both modes, including optimizer
updates and checkpoint round-trip. Optional offline integration tests use a tiny
random ESM to verify training/selection/evaluation control flow. Neither check
is a scientific benchmark.

**Full benchmark training has not been run: this execution environment has no
GPU. No held-out predictions, performance figures or fine-tuning improvement
claim exist yet.** Use the repository-backed Colab notebook to complete it.

This is one enzyme, one assay, one split and one seed. Random variant splitting
allows different substitutions at the same position across sets and assesses
interpolation within a nearly identical sequence family. It does not establish
transfer to unseen positions, multiple mutations, new enzymes, substrates or
purified-enzyme kinetics. Assay noise, expression-dependent activity and the
upstream notebook caveats remain. Fine-tuning may help, have no benefit, or hurt;
report the observed result without model-design or experimental-improvement claims.
Multiple seeds and position-held-out splits are possible follow-ups after this
fixed MVP, not substitutes for its first locked comparison.
