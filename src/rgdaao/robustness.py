"""Fixed-split, paired training-seed experiments and evidence-backed summaries."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config, validate_config, check_comparison
from .metrics import regression_metrics
from .prepare import prepare_dataset
from .split import assert_position_disjoint
from .training import assert_disjoint, file_sha256, write_json

MODES = ('frozen', 'finetune')
METRICS = ('spearman', 'pearson', 'mae', 'rmse')


def prepare_experiments(input_path: Path, output: Path, strategies=('position', 'random'),
                        seeds=(42, 43, 44), split_seed=42) -> dict:
    if not strategies or len(set(strategies)) != len(strategies) or set(strategies) - {'random', 'position'}:
        raise ValueError('Choose unique random/position strategies')
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError('At least two distinct training seeds are required')
    if any(type(s) is not int or not 0 <= s < 2**32 for s in (*seeds, split_seed)):
        raise ValueError('Seeds must be uint32 integers')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Refusing to overwrite robustness experiments: {output}')
    for name in ('wt.fasta', 'provenance.json'):
        if not (input_path.parent / name).is_file():
            raise ValueError(f'Missing source audit file: {name}')
    frozen = load_config('configs/esm2_frozen.yaml')
    tuned = load_config('configs/esm2_finetune.yaml')
    check_comparison(frozen, tuned)
    plan = {'schema_version': 1, 'split_seed': split_seed, 'training_seeds': list(seeds),
            'strategies': list(strategies), 'dataset_sha256': file_sha256(input_path),
            'jobs': [], 'splits': {}}
    for strategy in strategies:
        data_dir = output / 'data' / strategy
        manifest = prepare_dataset(input_path, data_dir, split_seed, strategy, list(seeds))
        provenance = json.loads((data_dir / 'provenance.json').read_text())
        if manifest['n_total'] != 6399 or not provenance['counts']['matches_paper_count']:
            raise ValueError('Robustness experiments require the complete published variant set')
        if provenance['dataset_sha256'] != plan['dataset_sha256']:
            raise ValueError('Source provenance does not match dataset')
        plan['splits'][strategy] = {'manifest': manifest,
                                   'manifest_sha256': file_sha256(data_dir / 'split_summary.json'),
                                   'provenance_sha256': file_sha256(data_dir / 'provenance.json'),
                                   'reference_sha256': file_sha256(data_dir / 'wt.fasta')}
        for seed in seeds:
            for mode, base in zip(MODES, (frozen, tuned)):
                config = deepcopy(base)
                config['training']['seed'] = seed
                config_path = f'configs/{strategy}_seed{seed}_{mode}.json'
                (output / 'configs').mkdir(exist_ok=True)
                write_json(output / config_path, config)
                plan['jobs'].append({'strategy': strategy, 'seed': seed, 'mode': mode,
                    'data_dir': f'data/{strategy}', 'run_dir': f'{strategy}/seed{seed}/esm2_{mode}',
                    'config_path': config_path, 'config_sha256': file_sha256(output / config_path)})
    write_json(output / 'plan.json', plan)
    validate_plan(output, plan)
    return plan


def validate_plan(output: Path, plan: dict) -> None:
    if plan['schema_version'] != 1:
        raise ValueError('Unsupported experiment plan')
    expected = {(s, n, m) for s in plan['strategies'] for n in plan['training_seeds'] for m in MODES}
    actual = [(j['strategy'], j['seed'], j['mode']) for j in plan['jobs']]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('Incomplete or duplicate paired experiment jobs')
    for strategy, record in plan['splits'].items():
        directory = output / 'data' / strategy
        for name, key in (('split_summary.json', 'manifest_sha256'),
                          ('provenance.json', 'provenance_sha256'), ('wt.fasta', 'reference_sha256')):
            if file_sha256(directory / name) != record[key]:
                raise ValueError(f'Changed {strategy} audit: {name}')
        manifest = record['manifest']
        if json.loads((directory / 'split_summary.json').read_text()) != manifest:
            raise ValueError('Planned manifest differs from saved split manifest')
        if (manifest['dataset_sha256'] != plan['dataset_sha256'] or
                manifest['seed'] != plan['split_seed'] or
                manifest['split_strategy'] != strategy or
                manifest['training_seeds'] != plan['training_seeds']):
            raise ValueError('Split manifest differs from planned experiment')
        # Audit split structure before any fitting, without using test labels
        # for tuning or running model predictions.
        frames = []
        for split in ('train', 'val', 'test'):
            path = directory / f'{split}.csv'
            if file_sha256(path) != manifest['split_sha256'][split]:
                raise ValueError(f'Changed {strategy}/{split} split')
            frame = pd.read_csv(path)
            if len(frame) != manifest[f'n_{split}']:
                raise ValueError('Split count mismatch')
            frames.append(frame)
        assert_disjoint(*frames)
        if strategy == 'position':
            assert_position_disjoint(*frames)
    pairs = {}
    for job in plan['jobs']:
        path = output / job['config_path']
        if file_sha256(path) != job['config_sha256']:
            raise ValueError('Changed planned training configuration')
        config = validate_config(json.loads(path.read_text()))
        if (config['training']['seed'] != job['seed'] or
                config['model']['freeze_backbone'] != (job['mode'] == 'frozen')):
            raise ValueError('Job identity differs from configuration')
        if (job['data_dir'] != f'data/{job["strategy"]}' or
                job['run_dir'] != f'{job["strategy"]}/seed{job["seed"]}/esm2_{job["mode"]}'):
            raise ValueError('Job paths differ from planned layout')
        pairs.setdefault((job['strategy'], job['seed']), {})[job['mode']] = config
    for pair in pairs.values():
        check_comparison(pair['frozen'], pair['finetune'])


def summarize_experiments(output: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    plan = json.loads((output / 'plan.json').read_text())
    validate_plan(output, plan)
    rows = []
    for job in plan['jobs']:
        directory = output / job['run_dir']
        run = json.loads((directory / 'run.json').read_text())
        if run['status'] != 'evaluated':
            raise ValueError('Summary requires every planned test evaluation to finish')
        if run['split_manifest'] != plan['splits'][job['strategy']]['manifest']:
            raise ValueError('Run used unplanned splits')
        if run['seed'] != job['seed'] or file_sha256(directory / 'config.json') != job['config_sha256']:
            raise ValueError('Run used unplanned training configuration')
        prediction_path = directory / 'predictions.csv'
        if file_sha256(prediction_path) != run['predictions_sha256']:
            raise ValueError('Predictions changed after evaluation')
        predictions = pd.read_csv(prediction_path)
        test = pd.read_csv(output / job['data_dir'] / 'test.csv')
        if (run['n_test'] != len(test) or not predictions.mutation.equals(test.mutation) or
                not np.allclose(predictions.experimental_activity, test.activity, rtol=0, atol=1e-12)):
            raise ValueError('Predictions do not match planned test identities and labels')
        metrics = regression_metrics(predictions.experimental_activity, predictions.predicted_activity)
        saved = json.loads((directory / 'metrics.json').read_text())
        for key, value in metrics.items():
            if value is None or not all(np.isclose(value, source[key]) for source in (saved, run['test_metrics'])):
                raise ValueError(f'Undefined or inconsistent metric: {key}')
        rows.append({k: job[k] for k in ('strategy', 'seed', 'mode')} | metrics)
    runs = pd.DataFrame(rows)
    aggregates, deltas = [], []
    for strategy in plan['strategies']:
        subset = runs[runs.strategy == strategy]
        for mode in MODES:
            group = subset[subset['mode'] == mode]
            for metric in METRICS:
                aggregates.append({'strategy': strategy, 'mode': mode, 'metric': metric,
                    'n_seeds': len(group), 'mean': group[metric].mean(), 'sd': group[metric].std(ddof=1)})
        for seed in plan['training_seeds']:
            pair = subset[subset.seed == seed].set_index('mode')
            for metric in METRICS:
                direction = 1 if metric in ('spearman', 'pearson') else -1
                deltas.append({'strategy': strategy, 'seed': seed, 'metric': metric,
                    'improvement': direction * (pair.loc['finetune', metric] - pair.loc['frozen', metric])})
    aggregate, paired = pd.DataFrame(aggregates), pd.DataFrame(deltas)
    delta_summary = paired.groupby(['strategy', 'metric']).improvement.agg(['count', 'mean', 'std']).reset_index()
    delta_summary = delta_summary.rename(columns={'count': 'n_seeds', 'std': 'sd'})
    for name, frame in [('runs.csv', runs), ('aggregate.csv', aggregate),
                        ('paired_deltas.csv', paired), ('paired_summary.csv', delta_summary)]:
        frame.to_csv(output / name, index=False)
    lines = ['# Robustness results', '',
             'Mean ± sample SD across training seeds on one fixed split per strategy.',
             'This SD measures training variation, not uncertainty across partitions or test samples.', '',
             '| Split | Model | Seeds | Spearman | Pearson | MAE | RMSE |',
             '|---|---|---:|---:|---:|---:|---:|']
    for strategy in plan['strategies']:
        for mode in MODES:
            g = runs[(runs.strategy == strategy) & (runs['mode'] == mode)]
            cells = [f'{g[m].mean():.3f} ± {g[m].std(ddof=1):.3f}' for m in METRICS]
            lines.append(f'| {strategy} | {mode} | {len(g)} | ' + ' | '.join(cells) + ' |')
    lines += ['', 'Positive paired improvements mean higher correlations or lower errors.',
              'No confidence interval or significance claim is inferred from three seeds.', '']
    (output / 'summary.md').write_text('\n'.join(lines))
    return runs, aggregate, paired
