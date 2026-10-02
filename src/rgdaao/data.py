from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from .mutations import apply_substitution, parse_mutation, validate_sequence

REQUIRED_COLUMNS = ('mutation', 'mutated_sequence', 'activity')


def load_variants(path: str | Path, wild_type: str | None = None) -> pd.DataFrame:
    """Load validated rows without dropping values or mutation identities."""
    df = pd.read_csv(path, float_precision='round_trip')
    missing = set(REQUIRED_COLUMNS) - set(df)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    columns = list(REQUIRED_COLUMNS) + (['expression'] if 'expression' in df else [])
    frame = df[columns].copy()
    if frame.empty or frame.isna().any().any():
        raise ValueError('Empty dataset or missing values')
    for column in ('mutation', 'mutated_sequence'):
        if frame[column].duplicated().any():
            raise ValueError(f'Duplicate {column} values')
    for column in columns[2:]:
        frame[column] = pd.to_numeric(frame[column], errors='raise')
        if not np.isfinite(frame[column]).all():
            raise ValueError(f'Non-finite {column} values')
    if frame.mutated_sequence.str.len().nunique() != 1:
        raise ValueError('Inconsistent sequence lengths')
    for row in frame.itertuples():
        validate_sequence(row.mutated_sequence)
        ref, pos, alt = parse_mutation(row.mutation)
        if pos > len(row.mutated_sequence) or row.mutated_sequence[pos - 1] != alt:
            raise ValueError(f'Mutated residue does not match {row.mutation}')
        reconstructed = row.mutated_sequence[:pos - 1] + ref + row.mutated_sequence[pos:]
        if wild_type is None:
            wild_type = reconstructed
        if apply_substitution(wild_type, row.mutation) != row.mutated_sequence:
            raise ValueError(f'Sequence inconsistent with reference for {row.mutation}')
    return frame.reset_index(drop=True)
