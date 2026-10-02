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
