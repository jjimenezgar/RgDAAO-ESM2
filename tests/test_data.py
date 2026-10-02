import pandas as pd
import pytest
from rgdaao.data import load_variants


def test_load_variants(tmp_path):
    path = tmp_path / "variants.csv"
    pd.DataFrame({
        "mutated_sequence": ["ACDE", "ACDF"],
        "activity": [1.0, 0.5],
        "extra": ["x", "y"],
    }).to_csv(path, index=False)
    df = load_variants(path)
    assert list(df.columns) == ["mutated_sequence", "activity"]
    assert len(df) == 2


def test_missing_columns_fail(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"sequence": ["ACDE"]}).to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_variants(path)
