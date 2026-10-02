from __future__ import annotations

import re

_MUTATION = re.compile(r"^([A-Z])(\d+)([A-Z])$")


def apply_substitution(wild_type: str, mutation: str) -> str:
    """Apply a substitution such as A42V to a wild-type protein sequence."""
    match = _MUTATION.fullmatch(mutation.strip().upper())
    if not match:
        raise ValueError(f"Expected substitution like A42V, got: {mutation}")
    ref, position, alt = match.groups()
    index = int(position) - 1
    if index < 0 or index >= len(wild_type):
        raise ValueError(f"Mutation position outside sequence: {mutation}")
    if wild_type[index] != ref:
        raise ValueError(
            f"Reference mismatch for {mutation}: sequence has {wild_type[index]} at {position}"
        )
    return wild_type[:index] + alt + wild_type[index + 1:]


def apply_substitutions(wild_type: str, mutations: str) -> str:
    """Apply colon/comma/semicolon-separated substitutions to a sequence."""
    sequence = wild_type.strip().upper()
    tokens = [x.strip() for x in re.split(r"[:,;]", mutations) if x.strip()]
    for token in tokens:
        sequence = apply_substitution(sequence, token)
    return sequence
