from __future__ import annotations

from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = ("mutated_sequence", "activity")


def load_variants(path: str | Path) -> pd.DataFrame:
    """Load and validate an RgDAAO variant table."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.loc[:, REQUIRED_COLUMNS].dropna().copy()
    out["mutated_sequence"] = out["mutated_sequence"].astype(str).str.strip().str.upper()
    out["activity"] = pd.to_numeric(out["activity"], errors="raise")

    if (out["mutated_sequence"].str.len() == 0).any():
        raise ValueError("Empty protein sequence found")
    if out["mutated_sequence"].duplicated().any():
        raise ValueError("Duplicate mutated_sequence values found")
    if not out["activity"].map(lambda x: pd.notna(x)).all():
        raise ValueError("Non-finite activity value found")
    return out.reset_index(drop=True)
