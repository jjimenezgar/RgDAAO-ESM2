from __future__ import annotations

import argparse
import json
import hashlib
import shutil
from pathlib import Path

from .data import load_variants
from .split import random_split, position_split, mutation_positions
from .source import read_fasta


def prepare_dataset(input_path: Path, out: Path, seed: int = 42,
                    strategy: str = 'random', training_seeds: list[int] | None = None) -> dict:
    if strategy not in ('random', 'position'):
        raise ValueError('Unknown split strategy')
    if training_seeds is not None and (not training_seeds or
            len(set(training_seeds)) != len(training_seeds) or
            any(type(s) is not int or not 0 <= s < 2**32 for s in training_seeds)):
        raise ValueError('Training seeds must be unique uint32 integers')
    reference = input_path.parent / 'wt.fasta'
    df = load_variants(input_path, read_fasta(reference) if reference.exists() else None)
    splitter = random_split if strategy == 'random' else position_split
    train, val, test = splitter(df, seed=seed)
    out.mkdir(parents=True, exist_ok=True)
    for name in ('wt.fasta', 'provenance.json'):
        source = input_path.parent / name
        if source.exists() and source.resolve() != (out / name).resolve():
            shutil.copyfile(source, out / name)
    distributions, hashes = {}, {}
    for name, frame in (('train', train), ('val', val), ('test', test)):
        path = out / f'{name}.csv'
        frame.to_csv(path, index=False, float_format='%.17g', lineterminator='\n')
        distributions[name] = frame.activity.describe().to_dict()
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    summary = {
        'n_total': len(df), 'n_train': len(train), 'n_val': len(val), 'n_test': len(test),
        'seed': seed, 'fractions': [0.8, 0.1, 0.1],
        'target_distributions': distributions, 'split_sha256': hashes,
        'dataset_sha256': hashlib.sha256(input_path.read_bytes()).hexdigest(),
    }
    # Preserve the original default manifest schema and original random split.
    if strategy != 'random' or training_seeds is not None:
        summary['split_strategy'] = strategy
        summary['fraction_unit'] = 'positions' if strategy == 'position' else 'variants'
        summary['position_sets'] = {n: sorted(mutation_positions(f)) for n, f in
                                    (('train', train), ('val', val), ('test', test))}
    if training_seeds is not None:
        summary['training_seeds'] = training_seeds
    (out / 'split_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and split the RgDAAO variant dataset.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--strategy', choices=['random', 'position'], default='random')
    parser.add_argument('--training-seeds', type=int, nargs='+', default=None)
    args = parser.parse_args()

    summary = prepare_dataset(Path(args.input), Path(args.output), args.seed,
                              args.strategy, args.training_seeds)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
