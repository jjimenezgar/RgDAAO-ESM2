from __future__ import annotations

import re
import pandas as pd

_SINGLE_AA = re.compile(r"^([A-Z])(\d+)([A-Z])$")


def canonical_single_missense(
    df: pd.DataFrame,
    mutation_column: str,
    activity_column: str,
    expression_column: str | None = None,
) -> pd.DataFrame:
    """Select single missense variants with experimental activity fitness.

    EP-Seq activity fitness is the direct log2 variant/WT activity score.
    This function deliberately does not substitute the later
    activity/expression-normalized score used for hotspot analysis.
    """
    required = [mutation_column, activity_column]
    if expression_column:
        required.append(expression_column)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing source columns: {missing}")

    out = df[required].dropna().copy()
    out["mutation"] = out[mutation_column].astype(str).str.strip().str.upper()
    out = out[out["mutation"].str.fullmatch(_SINGLE_AA)].copy()
    out["activity"] = pd.to_numeric(out[activity_column], errors="raise")
    if expression_column:
        out["expression"] = pd.to_numeric(out[expression_column], errors="raise")
    return out.reset_index(drop=True)
