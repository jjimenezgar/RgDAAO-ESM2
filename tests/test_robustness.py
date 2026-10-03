"""Synthetic workflow fixtures: these tests do not report enzyme performance."""
from copy import deepcopy
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

from rgdaao.prepare import prepare_dataset
from rgdaao.robustness import prepare_experiments, summarize_experiments, validate_plan
from rgdaao.split import assert_position_disjoint
from rgdaao.training import file_sha256, save_evaluation, write_json


@pytest.fixture(scope='module')
def source(tmp_path_factory):
    root = tmp_path_factory.mktemp('synthetic-source')
    wt = 'A' * 365
    rows = []
    for pos in range(1, 338):
        for alt in 'CDEFGHIKLMNPQRSTVWY':
            if len(rows) == 6399:
                break
            rows.append({'mutation': f'A{pos}{alt}',
                         'mutated_sequence': wt[:pos-1] + alt + wt[pos:],
                         'activity': pos / 400 + len(rows) % 19 / 100})
    path = root / 'variants.csv'
    pd.DataFrame(rows).sort_values('mutation').to_csv(path, index=False, float_format='%.17g')
    (root / 'wt.fasta').write_text('>synthetic\n' + wt + '\n')
    write_json(root / 'provenance.json', {'dataset_sha256': file_sha256(path),
               'counts': {'matches_paper_count': True}, 'synthetic_test_fixture': True})
    return path


@pytest.fixture(scope='module')
def prepared(source, tmp_path_factory):
    output = tmp_path_factory.mktemp('synthetic-plan')
    plan = prepare_experiments(source, output)
    return output, plan


def test_plan_fixed_partitions_and_paired_training_seeds(prepared):
    from rgdaao.config import load_config
    root, plan = prepared
    assert len(plan['jobs']) == 12
    validate_plan(root, plan)
    for job in plan['jobs']:
        config = load_config(root / job['config_path'])
        assert config['training']['learning_rate'] == 1e-5
    assert plan['training_seeds'] == [42, 43, 44]
    manifest = plan['splits']['position']['manifest']
    assert manifest['fraction_unit'] == 'positions'
    frames = [pd.read_csv(root / 'data/position' / f'{s}.csv') for s in ('train', 'val', 'test')]
    assert_position_disjoint(*frames)
    assert sum(map(len, frames)) == 6399
    assert len({j['data_dir'] for j in plan['jobs'] if j['strategy'] == 'position'}) == 1


def test_default_preparation_preserves_original_random_split(source, tmp_path):
    from rgdaao.data import load_variants
    from rgdaao.split import random_split
    manifest = prepare_dataset(source, tmp_path)
    assert 'training_seeds' not in manifest and 'split_strategy' not in manifest
    expected = random_split(load_variants(source), seed=42)
    for name, frame in zip(('train', 'val', 'test'), expected):
        saved = pd.read_csv(tmp_path / f'{name}.csv')
        assert saved.mutation.tolist() == frame.mutation.tolist()
    assert (tmp_path / 'wt.fasta').exists() and (tmp_path / 'provenance.json').exists()


def test_plan_rejects_changed_config_and_duplicate_job(prepared, tmp_path):
    source_root, plan = prepared
    root = tmp_path / 'copy'
    shutil.copytree(source_root, root)
    bad = deepcopy(plan)
    bad['jobs'].append(bad['jobs'][0])
    with pytest.raises(ValueError, match='duplicate'):
        validate_plan(root, bad)
    path = root / plan['jobs'][0]['config_path']
    path.write_text(path.read_text() + '\n')
    with pytest.raises(ValueError, match='configuration'):
        validate_plan(root, plan)


def write_synthetic_evaluations(root, plan):
    for job in plan['jobs']:
        directory = root / job['run_dir']
        directory.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / job['config_path'], directory / 'config.json')
        test = pd.read_csv(root / job['data_dir'] / 'test.csv')
        # Different nonconstant errors per model/seed exercise paired summaries.
        amplitude = (0.03 if job['mode'] == 'finetune' else 0.15) + (job['seed'] - 42) * 0.01
        predictions = test.activity + amplitude * np.sin(np.arange(len(test)))
        metrics = save_evaluation(test.mutation.tolist(), test.activity, predictions, directory)
        write_json(directory / 'run.json', {'status': 'evaluated', 'seed': job['seed'],
            'split_manifest': plan['splits'][job['strategy']]['manifest'],
            'n_test': len(test), 'test_metrics': metrics,
            'predictions_sha256': file_sha256(directory / 'predictions.csv')})


def test_summary_mean_sd_paired_errors_and_incomplete_rejection(prepared, tmp_path):
    source_root, plan = prepared
    root = tmp_path / 'copy'
    shutil.copytree(source_root, root)
    write_synthetic_evaluations(root, plan)
    runs, aggregate, deltas = summarize_experiments(root)
    assert len(runs) == 12 and len(aggregate) == 16 and len(deltas) == 24
    row = aggregate.query("strategy == 'position' and mode == 'finetune' and metric == 'mae'").iloc[0]
    values = runs.query("strategy == 'position' and mode == 'finetune'").mae
    assert row['mean'] == pytest.approx(values.mean())
    assert row['sd'] == pytest.approx(values.std(ddof=1))
    assert deltas.query("metric == 'mae'").improvement.gt(0).all()
    assert (root / 'summary.md').exists()
    job = plan['jobs'][0]
    path = root / job['run_dir'] / 'run.json'
    run = json.loads(path.read_text()); run['status'] = 'trained'; write_json(path, run)
    with pytest.raises(ValueError, match='every planned'):
        summarize_experiments(root)
    run['status'] = 'evaluated'; write_json(path, run)
    path = root / job['run_dir'] / 'predictions.csv'
    path.write_text(path.read_text() + '\n')
    with pytest.raises(ValueError, match='changed'):
        summarize_experiments(root)


def test_runner_trains_entire_suite_before_test_and_resumes(prepared, tmp_path, monkeypatch):
    import runpy
    import subprocess
    import sys
    source_root, plan = prepared
    root = tmp_path / 'copy'
    shutil.copytree(source_root, root)
    events = []

    def fake_run(command, **kwargs):
        assert kwargs['check'] is True
        if 'rgdaao.train' in command:
            directory = Path(command[command.index('--output-dir') + 1])
            job = next(j for j in plan['jobs'] if root / j['run_dir'] == directory)
            directory.mkdir(parents=True)
            shutil.copyfile(root / job['config_path'], directory / 'config.json')
            write_json(directory / 'run.json', {'status': 'trained', 'seed': job['seed'],
                'split_manifest': plan['splits'][job['strategy']]['manifest']})
            events.append('train')
        elif 'rgdaao.evaluate' in command:
            assert events.count('train') == len(plan['jobs'])
            directory = Path(command[command.index('--run-dir') + 1])
            job = next(j for j in plan['jobs'] if root / j['run_dir'] == directory)
            write_synthetic_evaluations(root, {'jobs': [job], 'splits': plan['splits']})
            events.append('evaluate')
        else:
            assert 'scripts/plot_results.py' in command
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, 'run', fake_run)
    monkeypatch.setattr(sys, 'argv', ['run_robustness.py', '--resume', '--output', str(root)])
    runpy.run_path('scripts/run_robustness.py', run_name='__main__')
    assert events == ['train'] * 12 + ['evaluate'] * 12
    events.clear()
    runpy.run_path('scripts/run_robustness.py', run_name='__main__')
    assert events == []  # Completed test evaluations are never repeated.


def test_colab_prepares_training_csv_before_smoke(source, tmp_path, monkeypatch):
    """Execute the actual plan cell with local preparation and a lightweight smoke probe."""
    import runpy
    import sys
    from rgdaao.data import load_variants
    from rgdaao.source import read_fasta
    repo_root = Path.cwd()
    notebook = json.loads((repo_root / 'notebooks/RgDAAO_ESM2_robustness.ipynb').read_text())
    cells = [''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code']
    assert 'scripts/smoke_test.py' not in cells[0]
    plan_cell = next(c for c in cells if 'STRATEGIES = ' in c)
    data = tmp_path / 'data/processed'; data.mkdir(parents=True)
    for name in ('variants.csv', 'provenance.json', 'wt.fasta'):
        shutil.copyfile(source.parent / name, data / name)
    shutil.copytree(repo_root / 'configs', tmp_path / 'configs')
    monkeypatch.chdir(tmp_path)
    events = []

    def run_visible(command):
        if command[1] == 'scripts/run_robustness.py':
            assert '--prepare-only' in command
            monkeypatch.setattr(sys, 'argv', command[1:])
            runpy.run_path(str(repo_root / command[1]), run_name='__main__')
            events.append('prepared')
        else:
            assert command[1] == 'scripts/smoke_test.py' and events == ['prepared']
            directory = Path(command[command.index('--data-dir') + 1])
            training = load_variants(directory / 'train.csv', read_fasta(directory / 'wt.fasta'))
            assert len(training) > 2
            test = pd.read_csv(directory / 'test.csv')
            assert set(training.head(2).mutation).isdisjoint(test.mutation)
            assert directory == Path('results/robustness/data/position')
            events.append('smoke')

    exec(compile(plan_cell, 'robustness-colab-plan-cell', 'exec'),
         {'Path': Path, 'sys': sys, 'run_visible': run_visible})
    assert events == ['prepared', 'smoke']
