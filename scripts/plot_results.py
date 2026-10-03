"""Plot completed real benchmark predictions; reject incomplete/smoke runs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from rgdaao.training import file_sha256
from rgdaao.metrics import regression_metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-dir', type=Path, default=Path('results'))
    args = p.parse_args()
    runs, predictions, rows = [], [], []
    for mode in ('frozen', 'finetune'):
        directory = args.results_dir / f'esm2_{mode}'
        run = json.loads((directory / 'run.json').read_text())
        if (run.get('status') != 'evaluated' or
                run.get('n_test') != run['split_manifest']['n_test'] or
                run['split_manifest']['n_total'] != 6399):
            raise ValueError('Figures require completed evaluation on the full held-out test set')
        if file_sha256(directory / 'predictions.csv') != run['predictions_sha256']:
            raise ValueError('Predictions changed after final evaluation')
        prediction = pd.read_csv(directory / 'predictions.csv')
        if len(prediction) != run['n_test'] or not prediction.mutation.is_unique:
            raise ValueError('Prediction count/identifiers disagree with the test manifest')
        metrics = regression_metrics(prediction.experimental_activity, prediction.predicted_activity)
        if any(not np.isclose(metrics[k], run['test_metrics'][k]) for k in metrics if metrics[k] is not None):
            raise ValueError('Saved metrics disagree with predictions')
        rows.append({'model': mode, **metrics})
        runs.append(run)
        predictions.append(prediction)
    if runs[0]['split_manifest']['split_sha256'] != runs[1]['split_manifest']['split_sha256']:
        raise ValueError('Models used different data partitions')
    if not predictions[0].mutation.equals(predictions[1].mutation):
        raise ValueError('Prediction identifiers differ')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42, 'savefig.dpi': 300})
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8), sharex=True, sharey=True, constrained_layout=True)
    all_values = np.concatenate([f[['experimental_activity', 'predicted_activity']].to_numpy().ravel() for f in predictions])
    lo, hi = all_values.min() - 0.03, all_values.max() + 0.03
    for ax, frame, row, title in zip(axes, predictions, rows, ('Frozen ESM-2', 'Fine-tuned ESM-2')):
        ax.scatter(frame.experimental_activity, frame.predicted_activity, s=10, alpha=0.35, color='#176C91', linewidths=0)
        ax.plot([lo, hi], [lo, hi], '--', color='0.45', linewidth=1)
        ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel='Experimental activity fitness',
               title=title + f"\nTest Spearman = {row['spearman']:.3f}" if row['spearman'] is not None else title + '\nSpearman undefined')
        ax.set_aspect('equal')
    axes[0].set_ylabel('Predicted activity fitness')
    out = args.results_dir / 'figures'
    out.mkdir(parents=True, exist_ok=True)
    for suffix in ('png', 'pdf'):
        fig.savefig(out / f'activity_predictions.{suffix}')
    plt.close(fig)
    pd.DataFrame(rows).to_csv(args.results_dir / 'comparison.csv', index=False)

    # Use epoch-end validation only: the last history record re-evaluates the
    # restored checkpoint and must not be mistaken for a new epoch-20 result.
    fig, ax = plt.subplots(figsize=(6.4, 3.8), constrained_layout=True)
    for mode, run, title, color in zip(
            ('frozen', 'finetune'), runs, ('Frozen ESM-2', 'Fine-tuned ESM-2'),
            ('#176C91', '#C45B28')):
        history = json.loads((args.results_dir / f'esm2_{mode}' / 'history.json').read_text())
        epoch_records = {}
        for record in history:
            if 'eval_spearman' in record:
                epoch_records.setdefault(record['step'], record)
        records = sorted(epoch_records.values(), key=lambda r: r['step'])
        ax.plot([r['epoch'] for r in records], [r['eval_spearman'] for r in records],
                label=title, color=color, linewidth=2)
        selected_step = int(run['best_checkpoint'].rsplit('-', 1)[1])
        selected = epoch_records[selected_step]
        ax.scatter(selected['epoch'], selected['eval_spearman'], color=color,
                   marker='*', s=140, zorder=3)
    ax.set(xlabel='Training epoch', ylabel='Validation Spearman',
           title='Validation ranking during training')
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    for suffix in ('png', 'pdf'):
        fig.savefig(out / f'validation_spearman.{suffix}')
    plt.close(fig)


if __name__ == '__main__':
    main()
