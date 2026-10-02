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


import io
import zipfile
import numpy as np
import pytest
from rgdaao.epseq import select_assay, build_epseq_dataset, FIGURE2, FIGURES6


def test_synonymous_is_excluded():
    frame = pd.DataFrame({'mut': ['A1A', 'A1V'], 'act': [0.1, 0.2]})
    assert canonical_single_missense(frame, 'mut', 'act').mutation.tolist() == ['A1V']


@pytest.mark.parametrize('mut,act', [('A1X', 0), ('nonsense text', 0), ('A0V', 0), ('A1V', np.inf), ('A1V', np.nan)])
def test_invalid_source_fails(mut, act):
    with pytest.raises(ValueError):
        canonical_single_missense(pd.DataFrame({'m': [mut], 'a': [act]}), 'm', 'a')


def test_duplicate_source_fails():
    with pytest.raises(ValueError, match='Duplicate'):
        canonical_single_missense(pd.DataFrame({'m': ['A1V', 'A1V'], 'a': [0.1, 0.2]}), 'm', 'a')


def test_tag_disagreement_fails():
    with pytest.raises(ValueError, match='tag disagrees'):
        select_assay(pd.DataFrame({'DAOx variant': ['A1A'], 'tag': ['missense'], 'score': [0]}), 'score')


def synthetic_archive(path, disagreement=False):
    # Synthetic unit-test measurements, never benchmark performance.
    activity = pd.DataFrame({'DAOx variant': ['C2W ', 'D3F', 'E4Y', 'C2C', 'D3*'],
                             'tag': ['missense']*3 + ['synonymous', 'nonsense'],
                             'activity  score': [0.1, -0.2, 0.3, 0, -0.5]})
    expression = pd.DataFrame({'DAOx variant': ['C2W', 'D3F', 'E4L'],
                               'tag': ['missense']*3, 'expression score': [0.2, 0.4, 0.5]})
    s6 = pd.DataFrame({'WT amino acid': ['C', 'D'], 'position': [2, 3], 'Mut amino acid': ['W', 'F'],
                       'activity fitness': [0.9 if disagreement else 0.1, -0.2], 'expression fitness': [0.2, 0.4]})
    a, b = io.BytesIO(), io.BytesIO()
    with pd.ExcelWriter(a, engine='openpyxl') as writer:
        activity.to_excel(writer, sheet_name='Fig2_H', index=False)
        expression.to_excel(writer, sheet_name='Fig2_F', index=False)
    with pd.ExcelWriter(b, engine='openpyxl') as writer:
        s6.to_excel(writer, sheet_name='FigS6_A', index=False)
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr(FIGURE2, a.getvalue())
        z.writestr(FIGURES6, b.getvalue())


def test_paired_workbook_join_and_reference(tmp_path):
    path = tmp_path / 'fixture.zip'
    synthetic_archive(path)
    frame, counts = build_epseq_dataset(path, 'ACDE')
    assert frame.mutation.tolist() == ['C2W', 'D3F']
    assert frame.mutated_sequence.tolist() == ['AWDE', 'ACFE']
    assert frame.activity.tolist() == [0.1, -0.2]
    assert counts['activity_only_missense'] == 1 and counts['expression_only_missense'] == 1
    assert not counts['matches_paper_count']  # never force the publication count


def test_independent_table_disagreement_fails(tmp_path):
    path = tmp_path / 'fixture.zip'
    synthetic_archive(path, disagreement=True)
    with pytest.raises(ValueError, match='disagree'):
        build_epseq_dataset(path, 'ACDE')
