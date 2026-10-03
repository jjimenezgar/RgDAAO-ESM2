"""One final held-out evaluation of a validation-selected checkpoint."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from .config import validate_config
from .data import load_variants
from .dataset import ProteinRegressionDataset
from .model import load_regression_model
from .source import read_fasta
from .split import assert_position_disjoint
from .training import assert_disjoint, file_sha256, save_evaluation, set_seed, write_json


def evaluate(run_dir: Path, data_dir: Path):
    import torch
    from torch.utils.data import DataLoader
    from transformers import DataCollatorWithPadding
    metadata = json.loads((run_dir / 'run.json').read_text())
    if metadata['status'] != 'trained' or (run_dir / 'predictions.csv').exists():
        raise ValueError('Require a completed training run with no previous test evaluation')
    config = validate_config(json.loads((run_dir / 'config.json').read_text()))
    if file_sha256(run_dir / 'best_model' / 'model.safetensors') != metadata['best_model_sha256']:
        raise ValueError('Selected checkpoint changed after validation selection')
    set_seed(config['training']['seed'])
    wt = read_fasta(data_dir / 'wt.fasta')
    frames = {}
    for name in ('train', 'val', 'test'):
        path = data_dir / f'{name}.csv'
        if file_sha256(path) != metadata['split_manifest']['split_sha256'][name]:
            raise ValueError(f'Changed {name} split')
        frames[name] = load_variants(path, wt)
    assert_disjoint(*frames.values())
    if metadata['split_manifest'].get('split_strategy') == 'position':
        assert_position_disjoint(*frames.values())
    test = frames['test']
    tokenizer, model = load_regression_model(
        freeze_backbone=config['model']['freeze_backbone'], checkpoint=run_dir / 'best_model')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    loader = DataLoader(ProteinRegressionDataset(test, tokenizer),
                        batch_size=config['training']['batch_size'], shuffle=False,
                        collate_fn=DataCollatorWithPadding(tokenizer))
    predictions = []
    with torch.no_grad():
        for batch in loader:
            batch.pop('labels')
            predictions.append(model(**{k: v.to(device) for k, v in batch.items()}).logits.cpu().numpy().reshape(-1))
    metrics = save_evaluation(test.mutation.tolist(), test.activity.to_numpy(), np.concatenate(predictions), run_dir)
    metadata.update(status='evaluated', test_metrics=metrics, n_test=len(test),
                    predictions_sha256=file_sha256(run_dir / 'predictions.csv'))
    write_json(run_dir / 'run.json', metadata)
    print(json.dumps({'experiment': str(run_dir), 'test': metrics}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=Path('data/processed'))
    args = parser.parse_args()
    evaluate(args.run_dir, args.data_dir)
