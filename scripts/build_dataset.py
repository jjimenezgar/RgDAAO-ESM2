"""Build the ML-ready RgDAAO single-missense activity dataset.

The source table/column names are command-line arguments on purpose: Zenodo
contains several processed outputs and we require an explicit provenance choice.

Example:
    python scripts/build_dataset.py \
      --table path/to/source.tsv \
      --wt data/raw/DAOx_ref.fa \
      --mutation-column MUTATION \
      --activity-column ACTIVITY_FITNESS \
      --output data/raw/rgdaao.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from rgdaao.epseq import canonical_single_missense
from rgdaao.source import read_fasta
from rgdaao.mutations import apply_substitution


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--table", required=True)
    p.add_argument("--wt", required=True)
    p.add_argument("--mutation-column", required=True)
    p.add_argument("--activity-column", required=True)
    p.add_argument("--expression-column")
    p.add_argument("--output", required=True)
    p.add_argument("--sep", default="\t")
    args = p.parse_args()

    source = pd.read_csv(args.table, sep=args.sep)
    variants = canonical_single_missense(
        source,
        args.mutation_column,
        args.activity_column,
        args.expression_column,
    )
    wt = read_fasta(args.wt)
    variants["mutated_sequence"] = variants["mutation"].map(
        lambda m: apply_substitution(wt, m)
    )

    cols = ["mutation", "mutated_sequence", "activity"]
    if "expression" in variants:
        cols.append("expression")
    out = variants[cols].drop_duplicates("mutation").reset_index(drop=True)

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(f"Wrote {len(out)} single-missense variants to {path}")


if __name__ == "__main__":
    main()
