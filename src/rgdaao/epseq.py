"""The explicit, source-verified EP-Seq phenotype selection."""
from __future__ import annotations
import io
import re
import zipfile
import numpy as np
import pandas as pd
from .mutations import parse_mutation, apply_substitution

FIGURE2 = 'Figures_data/2309_Figure_2.xlsx'
FIGURES6 = 'Figures_data/2309_Figure_S6.xlsx'
ACTIVITY_COLUMN = 'activity  score'  # two spaces in the deposited workbook


def canonical_single_missense(df, mutation_column, activity_column, expression_column=None):
    required = [mutation_column, activity_column] + ([expression_column] if expression_column else [])
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f'Missing source columns: {sorted(missing)}')
    if df[mutation_column].isna().any():
        raise ValueError('Missing mutation identifier')
    frame = df[required].copy()
    frame['mutation'] = frame[mutation_column].astype(str).str.strip()
    keep = []
    for mutation in frame.mutation:
        if mutation == 'WT':
            keep.append(False)
            continue
        tokens = re.split(r'[;,: ]+', mutation)
        if not all(re.fullmatch(r'[ACDEFGHIKLMNPQRSTVWY][1-9][0-9]*[ACDEFGHIKLMNPQRSTVWY*]', t) for t in tokens):
            raise ValueError(f'Unexpected mutation format: {mutation!r}')
        selected = len(tokens) == 1 and '*' not in mutation and mutation[0] != mutation[-1]
        if selected:
            parse_mutation(mutation)
        keep.append(selected)
    frame = frame.loc[keep].copy()
    if frame.mutation.duplicated().any():
        raise ValueError('Duplicate missense mutations')
    frame['activity'] = pd.to_numeric(frame[activity_column], errors='raise')
    columns = ['mutation', 'activity']
    if expression_column:
        frame['expression'] = pd.to_numeric(frame[expression_column], errors='raise')
        columns.append('expression')
    if not np.isfinite(frame[columns[1:]].to_numpy(dtype=float)).all():
        raise ValueError('Missing or non-finite experimental fitness')
    return frame[columns].reset_index(drop=True)


def select_assay(frame: pd.DataFrame, score: str) -> tuple[pd.DataFrame, dict]:
    required = {'DAOx variant', 'tag', score}
    if not required.issubset(frame):
        raise ValueError(f'Missing assay columns: {required - set(frame.columns)}')
    frame = frame.copy()
    if frame[list(required)].isna().any().any():
        raise ValueError('Missing assay values')
    frame['DAOx variant'] = frame['DAOx variant'].str.strip()
    frame['tag'] = frame['tag'].str.strip()
    if frame['DAOx variant'].duplicated().any():
        raise ValueError('Duplicate mutations in assay')
    if set(frame.tag) - {'missense', 'synonymous', 'nonsense'}:
        raise ValueError('Unexpected source variant tag')
    for mutation, tag in zip(frame['DAOx variant'], frame.tag):
        if not re.fullmatch(r'[ACDEFGHIKLMNPQRSTVWY][1-9][0-9]*[ACDEFGHIKLMNPQRSTVWY*]', mutation):
            raise ValueError(f'Unexpected source mutation: {mutation}')
        expected = 'nonsense' if mutation[-1] == '*' else ('synonymous' if mutation[0] == mutation[-1] else 'missense')
        if tag != expected:
            raise ValueError(f'Source tag disagrees with mutation: {mutation}, {tag}')
    if not np.isfinite(pd.to_numeric(frame[score], errors='raise')).all():
        raise ValueError('Non-finite assay fitness')
    counts = {str(k): int(v) for k, v in frame.tag.value_counts().items()}
    selected = canonical_single_missense(frame.loc[frame.tag == 'missense'], 'DAOx variant', score)
    return selected, counts


def build_epseq_dataset(archive_path, wild_type):
    with zipfile.ZipFile(archive_path) as archive:
        workbook = io.BytesIO(archive.read(FIGURE2))
        activity, activity_counts = select_assay(pd.read_excel(workbook, sheet_name='Fig2_H'), ACTIVITY_COLUMN)
        expression, expression_counts = select_assay(pd.read_excel(workbook, sheet_name='Fig2_F'), 'expression score')
        paired = pd.read_excel(io.BytesIO(archive.read(FIGURES6)), sheet_name='FigS6_A')
    merged = activity.merge(expression.rename(columns={'activity': 'expression'}), on='mutation', validate='one_to_one')
    if merged.empty:
        raise ValueError('No paired missense measurements')
    if not np.equal(paired.position, paired.position.astype(int)).all():
        raise ValueError('Non-integer source positions')
    paired['mutation'] = paired['WT amino acid'] + paired.position.astype(int).astype(str) + paired['Mut amino acid']
    paired = canonical_single_missense(paired, 'mutation', 'activity fitness', 'expression fitness')
    if set(merged.mutation) != set(paired.mutation):
        raise ValueError('Fig2 overlap differs from FigS6_A variant identities')
    check = merged.merge(paired, on='mutation', validate='one_to_one', suffixes=('', '_s6'))
    for label in ('activity', 'expression'):
        if not np.allclose(check[label], check[f'{label}_s6'], rtol=0, atol=1e-12):
            raise ValueError(f'Fig2 and FigS6_A disagree on {label}')
    merged['mutated_sequence'] = merged.mutation.map(lambda m: apply_substitution(wild_type, m))
    positions = merged.mutation.map(lambda m: parse_mutation(m)[1])
    if not positions.between(2, 365).all():
        raise ValueError('Unexpected mutation outside assayed DAOx region 2–365')
    merged = merged.sort_values('mutation').reset_index(drop=True)
    summary = {'activity_source_classes': activity_counts, 'expression_source_classes': expression_counts,
               'activity_only_missense': len(activity) - len(merged),
               'expression_only_missense': len(expression) - len(merged),
               'paired_missense': len(merged), 'expected_paper_count': 6399,
               'matches_paper_count': len(merged) == 6399,
               'supplemental_crosscheck': 'FigS6_A identities and both labels agree (atol=1e-12)'}
    return merged[['mutation', 'mutated_sequence', 'activity', 'expression']], summary
