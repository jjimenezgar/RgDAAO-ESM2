import pandas as pd
from rgdaao.epseq import canonical_single_missense


def test_keeps_only_single_missense():
    df = pd.DataFrame({
        "variant": ["A1V", "C2W", "A1V;C2W", "WT", "A1*"],
        "act": [0.2, -0.3, 0.1, 0.0, -1.0],
        "exp": [0.1, -0.2, 0.0, 0.0, -0.9],
    })
    out = canonical_single_missense(df, "variant", "act", "exp")
    assert out["mutation"].tolist() == ["A1V", "C2W"]
    assert out["activity"].tolist() == [0.2, -0.3]
