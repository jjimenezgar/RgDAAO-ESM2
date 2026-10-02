from __future__ import annotations

import numpy as np
import pandas as pd


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
