from __future__ import annotations
import re

AMINO_ACIDS = frozenset('ACDEFGHIKLMNPQRSTVWY')
_MUTATION = re.compile(r'^([ACDEFGHIKLMNPQRSTVWY])([1-9][0-9]*)([ACDEFGHIKLMNPQRSTVWY])$')


def parse_mutation(mutation: str) -> tuple[str, int, str]:
    """Strict one-based single missense notation; never renumber residues."""
    match = _MUTATION.fullmatch(mutation)
    if not match:
        raise ValueError(f'Expected single missense substitution like A42V, got: {mutation!r}')
    ref, position, alt = match.groups()
    if ref == alt:
        raise ValueError(f'Synonymous substitution is not missense: {mutation}')
    return ref, int(position), alt


def validate_sequence(sequence: str) -> None:
    if not isinstance(sequence, str) or not sequence or set(sequence) - AMINO_ACIDS:
        raise ValueError('Sequence must contain only the 20 standard uppercase amino acids')


def apply_substitution(wild_type: str, mutation: str) -> str:
    validate_sequence(wild_type)
    ref, position, alt = parse_mutation(mutation)
    index = position - 1
    if index >= len(wild_type):
        raise ValueError(f'Mutation position outside sequence: {mutation}')
    if wild_type[index] != ref:
        raise ValueError(f'Reference mismatch for {mutation}: sequence has {wild_type[index]} at {position}')
    return wild_type[:index] + alt + wild_type[index + 1:]


def apply_substitutions(wild_type: str, mutations: str) -> str:
    """Legacy helper; the EP-Seq benchmark accepts single substitutions only."""
    tokens = re.split(r'[:,;]', mutations)
    seen = set()
    sequence = wild_type
    for token in tokens:
        token = token.strip()
        _, position, _ = parse_mutation(token)
        if position in seen:
            raise ValueError(f'Repeated mutation position: {position}')
        seen.add(position)
        apply_substitution(wild_type, token)
        sequence = apply_substitution(sequence, token)
    return sequence
