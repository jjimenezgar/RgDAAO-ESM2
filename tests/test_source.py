import pandas as pd
from rgdaao.source import standardize_variant_table


def test_standardize_variant_table():
    source = pd.DataFrame({
        "variant": ["A1V", "C2W"],
        "activity_fitness": [0.8, -0.2],
    })
    result = standardize_variant_table(
        source, "ACDE", "variant", "activity_fitness"
    )
    assert result["mutated_sequence"].tolist() == ["VCDE", "AWDE"]
    assert result["activity"].tolist() == [0.8, -0.2]


import pytest
from rgdaao.source import read_fasta, read_rgdaao_reference


def test_multiple_fasta_records_rejected(tmp_path):
    path = tmp_path / 'bad.fa'
    path.write_text('>one\nACDE\n>two\nACDF\n')
    with pytest.raises(ValueError, match='exactly one'):
        read_fasta(path)


def test_construct_boundaries(tmp_path):
    # Synthetic construct with the documented coordinates; no downloaded data.
    dna = 'A'*905 + 'GGATCC' + 'ATG' + 'GCT'*364 + 'CTCGAG' + 'A'*94
    path = tmp_path / 'ref.fa'
    path.write_text('>synthetic\n'+dna+'\n')
    assert read_rgdaao_reference(path) == 'M' + 'A'*364
    path.write_text('>bad\n'+dna[:2006]+'AAAAAA'+dna[2012:]+'\n')
    with pytest.raises(ValueError, match='boundaries'):
        read_rgdaao_reference(path)
