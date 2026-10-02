import pytest
from rgdaao.mutations import apply_substitution, apply_substitutions


def test_apply_substitution():
    assert apply_substitution("ACDE", "C2W") == "AWDE"


def test_reference_mismatch_fails():
    with pytest.raises(ValueError, match="Reference mismatch"):
        apply_substitution("ACDE", "A2W")


def test_multiple_substitutions():
    assert apply_substitutions("ACDE", "A1V;E4F") == "VCDF"
