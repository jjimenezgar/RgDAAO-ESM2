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


def test_positions_stay_together_and_do_not_use_labels():
    from rgdaao.split import position_split, mutation_positions, assert_position_disjoint
    df = pd.DataFrame({'mutation': [f'A{i}{alt}' for i in range(1, 41) for alt in ('V', 'L')],
                       'activity': range(80)})
    a = position_split(df, seed=42)
    changed = df.copy()
    changed.activity = -changed.activity
    b = position_split(changed, seed=42)
    assert [f.mutation.tolist() for f in a] == [f.mutation.tolist() for f in b]
    assert [len(mutation_positions(f)) for f in a] == [32, 4, 4]
    assert_position_disjoint(*a)
    assert set().union(*(set(f.mutation) for f in a)) == set(df.mutation)
    for frame in a:
        assert frame.groupby(frame.mutation.str.extract(r'(\d+)')[0]).size().eq(2).all()
    assert not position_split(df, seed=43)[0].equals(a[0])


def test_position_leakage_rejected_even_with_distinct_variants():
    from rgdaao.split import assert_position_disjoint
    a = pd.DataFrame({'mutation': ['A10V', 'A11L']})
    b = pd.DataFrame({'mutation': ['A10W', 'A12L']})
    with pytest.raises(ValueError, match='positions'):
        assert_position_disjoint(a, b)


def test_position_split_rejects_too_few_groups_and_bad_notation():
    from rgdaao.split import position_split
    with pytest.raises(ValueError):
        position_split(pd.DataFrame({'mutation': ['A1V', 'A2L', 'A3L']}))
    with pytest.raises(ValueError, match='missense'):
        position_split(pd.DataFrame({'mutation': ['A1V:A2L']}))
