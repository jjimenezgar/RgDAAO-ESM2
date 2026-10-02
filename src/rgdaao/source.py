from __future__ import annotations
from pathlib import Path
from .mutations import apply_substitution, validate_sequence


def read_fasta(path: str | Path) -> str:
    lines = Path(path).read_text().splitlines()
    if not lines or not lines[0].startswith('>') or sum(s.startswith('>') for s in lines) != 1:
        raise ValueError('Expected exactly one FASTA record')
    sequence = ''.join(s.strip() for s in lines[1:])
    if not sequence:
        raise ValueError('No FASTA sequence found')
    return sequence


def read_rgdaao_reference(path: str | Path) -> str:
    """Translate construct DNA nts 912–2006 inclusive, one-based.

    Generate_LUT_notebook.ipynb cell 0 identifies DAOx start nt 912.
    The CDS ends before XhoI, consistent with the paper's BamHI/XhoI
    cloning and notebook DAOx positions 2–365.
    """
    from Bio.Seq import Seq
    dna = read_fasta(path)
    if len(dna) != 2106 or dna[905:911] != 'GGATCC' or dna[2006:2012] != 'CTCGAG':
        raise ValueError('Unexpected DAOx_ref.fa construct or coding boundaries')
    cds = dna[911:2006]
    if not cds.startswith('ATG') or set(cds) - set('ACGT'):
        raise ValueError('Invalid RgDAAO coding DNA')
    protein = str(Seq(cds).translate(table=1))
    validate_sequence(protein)
    if len(protein) != 365:
        raise ValueError('Expected 365-residue RgDAAO reference')
    return protein


def standardize_variant_table(source, wild_type, mutation_column, activity_column):
    """Legacy explicit-column adapter with single-substitution validation."""
    from .epseq import canonical_single_missense
    frame = canonical_single_missense(source, mutation_column, activity_column)
    frame['mutated_sequence'] = frame.mutation.map(lambda m: apply_substitution(wild_type, m))
    return frame[['mutation', 'mutated_sequence', 'activity']]
