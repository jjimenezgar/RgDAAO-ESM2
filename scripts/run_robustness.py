"""Prepare/run a paired multi-seed study; train all planned models before test evaluation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from rgdaao.robustness import prepare_experiments, summarize_experiments, validate_plan
from rgdaao.training import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('data/processed/variants.csv'))
    parser.add_argument('--output', type=Path, default=Path('results/robustness'))
    parser.add_argument('--strategies', nargs='+', choices=['random', 'position'], default=['position', 'random'])
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--split-seed', type=int, default=42)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Use the existing plan; skip completed stages')
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    if args.resume or args.summarize_only:
        plan = json.loads((args.output / 'plan.json').read_text())
        validate_plan(args.output, plan)
    else:
        plan = prepare_experiments(args.input, args.output, args.strategies, args.seeds, args.split_seed)
    if args.summarize_only:
        summarize_experiments(args.output)
        return
    print(f'Locked plan: {len(plan["jobs"])} model runs; split seed {plan["split_seed"]}; '
          f'training seeds {plan["training_seeds"]}', flush=True)
    if args.prepare_only:
        for strategy, record in plan['splits'].items():
            m = record['manifest']
            print(strategy, {n: m[f'n_{n}'] for n in ('train', 'val', 'test')}, flush=True)
        return
    # --resume deliberately reads settings from plan.json, not new CLI seeds.
    states = {}
    for job in plan['jobs']:
        run_path = args.output / job['run_dir'] / 'run.json'
        state = json.loads(run_path.read_text())['status'] if run_path.exists() else None
        if state not in (None, 'trained', 'evaluated'):
            raise ValueError(f'Interrupted training at {run_path}; preserve artifacts and use a fresh output directory')
        if state is not None:
            run = json.loads(run_path.read_text())
            if (run['seed'] != job['seed'] or
                    run['split_manifest'] != plan['splits'][job['strategy']]['manifest'] or
                    file_sha256(run_path.parent / 'config.json') != job['config_sha256']):
                raise ValueError('Completed run differs from locked experiment plan')
        states[job['run_dir']] = state
    # An interrupted evaluation stage can resume only after every model trained.
    if 'evaluated' in states.values() and any(s is None for s in states.values()):
        raise ValueError('Cannot fit additional models after test evaluation has started')
    for job in plan['jobs']:
        if states[job['run_dir']] is not None:
            continue
        env = dict(os.environ, PYTHONHASHSEED=str(job['seed']), CUBLAS_WORKSPACE_CONFIG=':4096:8')
        print(f'Train {job["strategy"]} seed={job["seed"]} {job["mode"]}', flush=True)
        subprocess.run([sys.executable, '-m', 'rgdaao.train',
            '--config', str(args.output / job['config_path']),
            '--data-dir', str(args.output / job['data_dir']),
            '--output-dir', str(args.output / job['run_dir'])], check=True, env=env)
    for job in plan['jobs']:
        if states[job['run_dir']] == 'evaluated':
            continue
        subprocess.run([sys.executable, '-m', 'rgdaao.evaluate',
            '--run-dir', str(args.output / job['run_dir']),
            '--data-dir', str(args.output / job['data_dir'])], check=True)
    summarize_experiments(args.output)
    for strategy in plan['strategies']:
        for seed in plan['training_seeds']:
            subprocess.run([sys.executable, 'scripts/plot_results.py', '--results-dir',
                str(args.output / strategy / f'seed{seed}')], check=True)
    print((args.output / 'summary.md').read_text(), flush=True)


if __name__ == '__main__':
    main()
