# Data

Raw experimental files are not committed to this repository.

## Source

Vanella et al. (2024), *Nature Communications* 15, 1807.
Zenodo: https://doi.org/10.5281/zenodo.8388902

The Zenodo record contains raw sequencing data as well as much smaller processed
figure/fitness resources. Reproducing the ML benchmark does not require
downloading the 14.3 GB Illumina archive unless the goal is to reproduce the
experimental sequencing analysis itself.

Relevant deposited resources include:

- `DAOx_ref.fa` — reference sequence;
- `Figures_data.zip` — processed data underlying figures;
- `Jupyter_notebooks_fitness_calculation.zip` — fitness-calculation notebooks/data;
- `Look_up_table.tsv` — variant lookup table;
- raw PacBio and Illumina sequencing files.

Run `python scripts/inspect_zenodo.py` to inspect the current Zenodo file list.

## Benchmark policy

Before generating `data/processed/*.csv`, the exact source file, source
columns, filtering rule and transformation must be recorded here. We do not
silently infer column meanings or train on raw sequencing counts.
