# RgDAAO-ESM2

A small, reproducible benchmark for adapting ESM-2 to predict experimentally measured mutational effects on *Rhodotorula gracilis* D-amino acid oxidase (RgDAAO).

## Scope

This repository does **not** introduce a new protein language model or claim a new enzyme-engineering method. Its purpose is to compare two established transfer-learning strategies on an enzyme deep-mutational-scanning dataset:

1. **Frozen ESM-2** — extract sequence representations and train only a regression head.
2. **Fine-tuned ESM-2** — update part or all of the pretrained model together with the regression head.

The primary target is experimental enzymatic activity. Model predictions are computational estimates and are not substitutes for experimental validation.

## Biological system

RgDAAO is a flavoprotein D-amino acid oxidase from *Rhodotorula gracilis*. Structural interpretation can use PDB **1C0P**, which contains RgDAAO with FAD and a bound active-site ligand.

## Planned benchmark

- Experimental missense-variant data
- Reproducible train/validation/test partitions
- Frozen ESM-2 baseline
- ESM-2 fine-tuning with PyTorch
- Pearson and Spearman correlation, MAE and RMSE
- Saved predictions and machine-readable metrics
- Optional structural mapping of prediction errors after the core benchmark is validated

## Status

**V0.1 — pipeline scaffold.** Data validation, deterministic splitting, regression metrics, configuration and tests are implemented. No benchmark numbers are reported yet: results will only be added after training on the documented experimental dataset.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Prepare a CSV with at least:

```text
mutated_sequence,activity
...
```

Then validate and split it:

```bash
python -m rgdaao.prepare --input data/raw/rgdaao.csv --output data/processed
```

Training code for ESM-2 is intentionally kept separate from CI because model weights and GPU training are comparatively expensive.

## Reproducibility principles

- fixed random seeds;
- train/validation/test separation before model fitting;
- no test-set model selection;
- explicit dataset provenance;
- machine-readable outputs;
- no biological claim from model predictions alone.

## References

Dataset provenance and the exact experimental publication will be documented alongside the data-preparation step before benchmark results are released.
