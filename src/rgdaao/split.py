from __future__ import annotations

import numpy as np
import pandas as pd
from .mutations import parse_mutation


def mutation_positions(frame: pd.DataFrame) -> set[int]:
    return {parse_mutation(m)[1] for m in frame.mutation}


def assert_position_disjoint(*frames: pd.DataFrame) -> None:
    seen = set()
    for frame in frames:
        positions = mutation_positions(frame)
        if seen & positions:
            raise ValueError('Split leakage: overlapping mutation positions')
        seen.update(positions)


def position_split(df: pd.DataFrame, train_fraction: float = 0.8,
                   val_fraction: float = 0.1, seed: int = 42):
    """Allocate whole residue positions; fractions apply to groups, not rows.

    Positions are sorted numerically before permutation. No activity labels
    are used to select groups; row order is retained within each partition.
    """
    positions = np.array(sorted(mutation_positions(df)))
    groups = random_split(pd.DataFrame({'position': positions}),
                          train_fraction, val_fraction, seed)
    row_positions = df.mutation.map(lambda m: parse_mutation(m)[1])
    frames = tuple(df.loc[row_positions.isin(g.position)].reset_index(drop=True) for g in groups)
    if min(map(len, frames)) < 2:
        raise ValueError('Each split requires at least two observations for correlation')
    assert_position_disjoint(*frames)
    return frames


def random_split(
    df: pd.DataFrame,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Deterministic random split used as the initial benchmark split."""
    if not (0 < train_fraction < 1 and 0 < val_fraction < 1):
        raise ValueError("Invalid split fractions")
    if train_fraction + val_fraction >= 1:
        raise ValueError("train_fraction + val_fraction must be < 1")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(df))
    n_train = int(len(df) * train_fraction)
    n_val = int(len(df) * val_fraction)
    if min(n_train, n_val, len(df) - n_train - n_val) < 2:
        raise ValueError("Each split requires at least two observations for correlation")

    train = df.iloc[order[:n_train]].reset_index(drop=True)
    val = df.iloc[order[n_train:n_train + n_val]].reset_index(drop=True)
    test = df.iloc[order[n_train + n_val:]].reset_index(drop=True)
    return train, val, test
