from __future__ import annotations

from pathlib import Path
import pandas as pd

from .mutations import apply_substitutions


def read_fasta(path: str | Path) -> str:
    lines = Path(path).read_text().splitlines()
    sequence = "".join(line.strip() for line in lines if line and not line.startswith(">"))
    if not sequence:
        raise ValueError("No FASTA sequence found")
    return sequence.upper()


def standardize_variant_table(
    source: pd.DataFrame,
    wild_type: str,
    mutation_column: str,
    activity_column: str,
) -> pd.DataFrame:
    """Convert a documented source table to the benchmark's two-column schema."""
    if mutation_column not in source or activity_column not in source:
        raise ValueError("Requested mutation/activity columns are absent from source table")

    frame = source[[mutation_column, activity_column]].dropna().copy()
    frame["mutated_sequence"] = frame[mutation_column].astype(str).map(
        lambda mutation: apply_substitutions(wild_type, mutation)
    )
    frame["activity"] = pd.to_numeric(frame[activity_column], errors="raise")
    return frame[["mutated_sequence", "activity"]].reset_index(drop=True)
