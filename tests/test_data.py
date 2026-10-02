import numpy as np
import pandas as pd
import pytest
from rgdaao.data import load_variants


def example():
    return pd.DataFrame({'mutation': ['A1V', 'C2W'], 'mutated_sequence': ['VCDE', 'AWDE'],
                         'activity': [1.0, 0.5], 'expression': [0.1, 0.2]})


def test_load_variants_preserves_identifiers(tmp_path):
    path = tmp_path / 'variants.csv'
    example().assign(extra='unused').to_csv(path, index=False)
    df = load_variants(path, 'ACDE')
    assert list(df.columns) == ['mutation', 'mutated_sequence', 'activity', 'expression']
    assert df.mutation.tolist() == ['A1V', 'C2W']


@pytest.mark.parametrize('column,value', [
    ('activity', np.inf), ('activity', np.nan), ('expression', -np.inf),
    ('mutated_sequence', 'ACXE'), ('mutated_sequence', 'VCDEF'),
    ('mutated_sequence', 'VCDF'), ('mutation', 'A0V'), ('mutation', 'A1A'),
    ('mutation', 'C1V'), ('mutation', 'A1W'), ('mutation', 'Z1V')])
def test_invalid_rows_fail(tmp_path, column, value):
    frame = example()
    frame.loc[0, column] = value
    path = tmp_path / 'bad.csv'
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_variants(path, 'ACDE')


def test_missing_columns_and_duplicates_fail(tmp_path):
    path = tmp_path / 'bad.csv'
    for frame in (example().drop(columns='mutation'), pd.concat([example(), example()])):
        frame.to_csv(path, index=False)
        with pytest.raises(ValueError):
            load_variants(path)
