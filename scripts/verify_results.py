"""Verify the committed benchmark evidence without weights, GPU or test re-evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rgdaao.config import check_comparison, validate_config
from rgdaao.metrics import regression_metrics
from rgdaao.training import file_sha256


def verify(results_dir: Path) -> None:
    def read(name):
        return json.loads((results_dir / name).read_text())

    def require(condition, message):
        if not condition:
            raise ValueError(message)

    audit = read('import_audit.json')
    for name, expected in audit['source_files_sha256'].items():
        require(file_sha256(results_dir / name) == expected, f'Imported artifact changed: {name}')
    manifest = read('data_audit/split_summary.json')
    provenance = read('data_audit/provenance.json')
    require(manifest['dataset_sha256'] == provenance['dataset_sha256'], 'Dataset hash mismatch')
    comparison = pd.read_csv(results_dir / 'comparison.csv').set_index('model')
    frames, configs = [], []
    for mode in ('frozen', 'finetune'):
        prefix = f'esm2_{mode}/'
        run = read(prefix + 'run.json')
        environment = read(prefix + 'environment.json')
        config = validate_config(read(prefix + 'config.json'))
        configs.append(config)
        require(run['status'] == 'evaluated', 'Incomplete run')
        require(run['split_manifest'] == manifest, 'Run used different splits')
        require(run['dataset_provenance'] == provenance, 'Run used different data')
        require(run['seed'] == config['training']['seed'] == manifest['seed'], 'Seed mismatch')
        require(environment['git_commit'] == audit['training_git_commit'], 'Training commit mismatch')
        require(environment['git_dirty'] is False, 'Training code was modified')
        path = results_dir / prefix / 'predictions.csv'
        require(file_sha256(path) == run['predictions_sha256'], 'Prediction hash mismatch')
        frame = pd.read_csv(path)
        require(len(frame) == run['n_test'] == manifest['n_test'], 'Test count mismatch')
        require(frame.mutation.is_unique, 'Duplicate test variants')
        metrics = regression_metrics(frame.experimental_activity, frame.predicted_activity)
        for saved in (read(prefix + 'metrics.json'), run['test_metrics'], comparison.loc[mode]):
            for key, value in metrics.items():
                require(value is not None and np.isclose(value, saved[key], rtol=1e-7, atol=1e-9),
                        f'{mode}: {key} disagrees with predictions')
        history = read(prefix + 'history.json')
        evaluations = [h for h in history if 'eval_spearman' in h]
        best = max(h['eval_spearman'] for h in evaluations)
        require(np.isclose(best, run['best_validation_spearman']), 'Wrong validation selection')
        selected_step = int(run['best_checkpoint'].rsplit('-', 1)[1])
        selected = [h for h in evaluations if h['step'] == selected_step]
        require(any(np.isclose(h['eval_spearman'], best) for h in selected), 'Wrong checkpoint step')
        frames.append(frame)
    check_comparison(*configs)
    require(frames[0].mutation.equals(frames[1].mutation), 'Models predict different variants')
    require(np.array_equal(frames[0].experimental_activity, frames[1].experimental_activity),
            'Models evaluated different labels')
    print(f'Verified both seed-{manifest["seed"]} runs: {manifest["n_test"]} test variants, '
          'original artifact hashes, metrics and validation selection.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'docs/results/seed42')
    verify(parser.parse_args().results_dir)


if __name__ == '__main__':
    main()
