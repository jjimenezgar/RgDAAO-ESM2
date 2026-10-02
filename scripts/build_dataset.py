"""Build the documented paired single-missense EP-Seq activity benchmark."""
from __future__ import annotations
import argparse
import hashlib
import json
import warnings
from pathlib import Path
from download_data import FILES, matches
from rgdaao.epseq import build_epseq_dataset, FIGURE2, ACTIVITY_COLUMN
from rgdaao.source import read_rgdaao_reference


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir', type=Path, default=Path('data/source'))
    p.add_argument('--output', type=Path, default=Path('data/processed'))
    args = p.parse_args()
    for name in ('DAOx_ref.fa', 'Figures_data.zip'):
        if not matches(args.source_dir / name, *FILES[name]):
            raise ValueError(f'Unverified source {name}; run scripts/download_data.py')
    wt = read_rgdaao_reference(args.source_dir / 'DAOx_ref.fa')
    frame, counts = build_epseq_dataset(args.source_dir / 'Figures_data.zip', wt)
    if not counts['matches_paper_count']:
        warnings.warn(f'Paper reports 6399 paired missense variants; obtained {len(frame)}. Investigate before training.')
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / 'variants.csv'
    frame.to_csv(path, index=False, float_format='%.17g', lineterminator='\n')
    (args.output / 'wt.fasta').write_text('>RgDAAO_EPSeq_365aa_DAOx_ref_nt912-2006\n' + wt + '\n')
    provenance = {
        'zenodo_record': '8388902', 'paper_doi': '10.1038/s41467-024-45630-3',
        'source_file': f'Figures_data.zip::{FIGURE2}',
        'activity': {'sheet': 'Fig2_H', 'column': ACTIVITY_COLUMN},
        'expression': {'sheet': 'Fig2_F', 'column': 'expression score', 'role': 'auxiliary only'},
        'filtering': 'tag=missense in each assay; inner join on stripped DAOx variant; validate all mutations',
        'label_transformations': 'none; deposited consensus log2 fitness relative to WT',
        'mutation_notation': 'one-based RgDAAO A123V; no 104-residue offset applied to figure tables',
        'wt_sequence': wt, 'sequence_length': len(wt),
        'reference': 'DAOx_ref.fa DNA, nts 912–2006 inclusive, standard genetic code; BamHI/XhoI boundaries',
        'source_sha256': {n: hashlib.sha256((args.source_dir / n).read_bytes()).hexdigest()
                          for n in ('DAOx_ref.fa', 'Figures_data.zip')},
        'dataset_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'counts': counts,
    }
    (args.output / 'provenance.json').write_text(json.dumps(provenance, indent=2, allow_nan=False) + '\n')
    print(json.dumps(counts, indent=2))


if __name__ == '__main__':
    main()
