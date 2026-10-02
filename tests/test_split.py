import pandas as pd
from rgdaao.split import random_split


def test_split_is_deterministic_and_disjoint():
    df = pd.DataFrame({
        "mutated_sequence": [f"SEQ{i}" for i in range(100)],
        "activity": range(100),
    })
    a = random_split(df, seed=7)
    b = random_split(df, seed=7)
    assert [x["mutated_sequence"].tolist() for x in a] == [x["mutated_sequence"].tolist() for x in b]

    sets = [set(x["mutated_sequence"]) for x in a]
    assert sets[0].isdisjoint(sets[1])
    assert sets[0].isdisjoint(sets[2])
    assert sets[1].isdisjoint(sets[2])
    assert sum(map(len, a)) == len(df)


import pytest


@pytest.mark.parametrize('train,val', [(0.9, 0.1), (0.8, 0), (1.2, 0.1), (0.8, float('nan'))])
def test_invalid_split_fractions(train, val):
    with pytest.raises(ValueError):
        random_split(pd.DataFrame({'x': range(100)}), train, val)


def test_real_dataset_split_sizes():
    splits = random_split(pd.DataFrame({'id': range(6399)}), seed=42)
    assert [len(x) for x in splits] == [5119, 639, 641]
    other = random_split(pd.DataFrame({'id': range(6399)}), seed=43)
    assert not splits[0].equals(other[0])
