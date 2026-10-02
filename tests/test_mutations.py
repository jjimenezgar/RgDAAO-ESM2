import pytest
from rgdaao.mutations import apply_substitution, apply_substitutions


def test_apply_substitution():
    assert apply_substitution("ACDE", "C2W") == "AWDE"


def test_reference_mismatch_fails():
    with pytest.raises(ValueError, match="Reference mismatch"):
        apply_substitution("ACDE", "A2W")


def test_multiple_substitutions():
    assert apply_substitutions("ACDE", "A1V;E4F") == "VCDF"


@pytest.mark.parametrize('mutation', ['A0V', 'A5V', 'A01V', 'A1A', 'A1*', 'A1X', 'B1V', 'a1v', 'A1V;C2W', ' A1V'])
def test_invalid_substitutions_fail(mutation):
    with pytest.raises(ValueError):
        apply_substitution('ACDE', mutation)


def test_length_and_changed_residue():
    wt = 'ACDEFGHIKLMNPQRSTVWY'
    result = apply_substitution(wt, 'E4A')
    assert len(result) == len(wt) and result[3] == 'A'
    assert sum(a != b for a, b in zip(wt, result)) == 1


def test_repeated_position_fails():
    with pytest.raises(ValueError, match='Repeated'):
        apply_substitutions('ACDE', 'A1V;V1L')
