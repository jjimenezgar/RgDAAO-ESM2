"""Train/select on train and validation only; evaluate.py opens the test set later."""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path
from .config import load_config
from .data import load_variants
from .dataset import ProteinRegressionDataset
from .metrics import regression_metrics
from .model import load_regression_model, trainable_parameters, make_optimizer
from .source import read_fasta
from .split import assert_position_disjoint
from .training import set_seed, write_json, environment, file_sha256, assert_disjoint


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    metrics = regression_metrics(labels, predictions.reshape(-1))
    if metrics['spearman'] is None:
        raise ValueError('Undefined validation Spearman; no valid checkpoint can be selected')
    return metrics


def train(config, data_dir: Path, output: Path):
    from transformers import DataCollatorWithPadding, Trainer, TrainingArguments
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Refusing to overwrite experiment: {output}')
    set_seed(config['training']['seed'])
    summary = json.loads((data_dir / 'split_summary.json').read_text())
    provenance = json.loads((data_dir / 'provenance.json').read_text())
    if config['training']['seed'] not in summary.get('training_seeds', [summary['seed']]):
        raise ValueError('Configuration seed is not authorized by the split manifest')
    if not provenance['counts']['matches_paper_count']:
        raise ValueError('Resolve the dataset count discrepancy before running this benchmark')
    if provenance['dataset_sha256'] != summary['dataset_sha256']:
        raise ValueError('Dataset provenance does not match split manifest')
    wt = read_fasta(data_dir / 'wt.fasta')
    frames = {}
    for name in ('train', 'val'):
        path = data_dir / f'{name}.csv'
        if file_sha256(path) != summary['split_sha256'][name]:
            raise ValueError(f'Changed {name} split; regenerate or investigate')
        frames[name] = load_variants(path, wt)
    assert_disjoint(*frames.values())
    if summary.get('split_strategy') == 'position':
        assert_position_disjoint(*frames.values())
    tokenizer, model = load_regression_model(config['model']['name'], config['model']['freeze_backbone'], config['model']['revision'])
    trainable, total = trainable_parameters(model)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'config.json', config)
    write_json(output / 'environment.json', environment())
    (output / 'pip-freeze.txt').write_text(subprocess.check_output(
        [sys.executable, '-m', 'pip', 'freeze'], text=True))
    metadata = {'status': 'training', 'seed': config['training']['seed'],
                'model_name': config['model']['name'], 'model_revision': config['model']['revision'],
                'trainable_parameters': trainable, 'total_parameters': total,
                'split_manifest': summary, 'dataset_provenance': provenance}
    write_json(output / 'run.json', metadata)
    t = config['training']
    args = TrainingArguments(
        output_dir=str(output / 'checkpoints'),
        learning_rate=t['learning_rate'], per_device_train_batch_size=t['batch_size'],
        per_device_eval_batch_size=t['batch_size'], num_train_epochs=t['epochs'],
        weight_decay=t['weight_decay'], eval_strategy='epoch', save_strategy='epoch',
        load_best_model_at_end=True, metric_for_best_model='spearman', greater_is_better=True,
        save_total_limit=1, seed=t['seed'], data_seed=t['seed'], full_determinism=True,
        report_to=[], optim='adamw_torch', lr_scheduler_type='linear', warmup_ratio=0.0,
        dataloader_num_workers=0, logging_strategy='epoch', fp16=False, bf16=False)
    trainer = Trainer(model=model, args=args,
        train_dataset=ProteinRegressionDataset(frames['train'], tokenizer),
        eval_dataset=ProteinRegressionDataset(frames['val'], tokenizer),
        data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=compute_metrics,
        optimizers=(make_optimizer(model, t), None))
    trainer.train()
    # Trainer has restored the highest-validation-Spearman checkpoint.
    validation = trainer.evaluate()
    trainer.save_model(output / 'best_model')
    tokenizer.save_pretrained(output / 'best_model')
    metadata.update(status='trained', best_validation=validation,
                    best_validation_spearman=trainer.state.best_metric,
                    best_checkpoint=Path(trainer.state.best_model_checkpoint).name,
                    best_model_sha256=file_sha256(output / 'best_model' / 'model.safetensors'))
    write_json(output / 'run.json', metadata)
    write_json(output / 'history.json', trainer.state.log_history)
    print(json.dumps({'status': 'trained; test not evaluated', 'best_validation': validation}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--data-dir', type=Path, default=Path('data/processed'))
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    train(load_config(args.config), args.data_dir, args.output_dir)


if __name__ == '__main__':
    main()
