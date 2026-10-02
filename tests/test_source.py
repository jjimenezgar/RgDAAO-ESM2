import pandas as pd
from rgdaao.source import standardize_variant_table


def test_standardize_variant_table():
    source = pd.DataFrame({
        "variant": ["A1V", "C2W"],
        "activity_fitness": [0.8, -0.2],
    })
    result = standardize_variant_table(
        source, "ACDE", "variant", "activity_fitness"
    )
    assert result["mutated_sequence"].tolist() == ["VCDE", "AWDE"]
    assert result["activity"].tolist() == [0.8, -0.2]
