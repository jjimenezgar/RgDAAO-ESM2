# Benchmark definition

## Primary prediction task

Predict the **experimental EP-Seq activity fitness** of single missense RgDAAO
variants from protein sequence.

The Nature Communications study defines variant fitness relative to wild type
using a log2 variant/WT score. For the core ML benchmark we use the direct
activity-fitness phenotype for the 6,399 single missense variants measured in
both activity and expression assays.

We do **not** use the later activity/expression-normalized quantity as the
primary label. That normalization was introduced by the authors to identify
activity-enhancing hotspots while reducing expression/folding effects and is a
different biological question.

## Substrate

The activity branch of EP-Seq assays RgDAAO catalysis using D-alanine and
detects the H2O2-producing oxidase reaction through proximity labeling.

## Models

The first comparison is intentionally narrow:

- frozen ESM-2 + regression head;
- fine-tuned ESM-2 + regression head.

Both models must use the same deterministic data partitions and evaluation
metrics. The held-out test set is used once for final comparison.

## Metrics

Primary: Spearman correlation.

Also reported: Pearson correlation, MAE and RMSE.

## Source

Vanella R. et al. (2024), Nature Communications 15, 1807.
DOI: 10.1038/s41467-024-45630-3
Zenodo record: 8388902.
