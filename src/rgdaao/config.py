"""Small, strict configuration checks; no ML imports."""
from __future__ import annotations
import math
import json
from pathlib import Path
import yaml

MODEL_NAME = 'facebook/esm2_t6_8M_UR50D'
MODEL_REVISION = 'c731040fcd8d73dceaa04b0a8e6329b345b0f5df'


def validate_config(config: dict) -> dict:
    if not isinstance(config, dict) or set(config) != {'model', 'training', 'target'}:
        raise ValueError('Configuration must contain model, training and target')
    model, training = config['model'], config['training']
    if not isinstance(model, dict) or not isinstance(training, dict):
        raise ValueError('model and training must be mappings')
    if set(model) != {'name', 'revision', 'freeze_backbone'}:
        raise ValueError('Expected model name, revision and freeze_backbone')
    if model['name'] != MODEL_NAME or model['revision'] != MODEL_REVISION:
        raise ValueError('This benchmark uses the pinned ESM-2 8M checkpoint')
    if type(model['freeze_backbone']) is not bool:
        raise ValueError('freeze_backbone must be a YAML boolean, not a string')
    if config['target'] != 'activity':
        raise ValueError('The primary target must be direct activity fitness')
    required = {'seed', 'learning_rate', 'head_learning_rate', 'epochs', 'batch_size', 'weight_decay'}
    if set(training) != required:
        raise ValueError(f'Expected training keys: {sorted(required)}')
    for key in ('seed', 'epochs', 'batch_size'):
        value = training[key]
        if type(value) is not int or value < (0 if key == 'seed' else 1):
            raise ValueError(f'Invalid {key}')
    if training['seed'] >= 2**32:
        raise ValueError('Seed must be less than 2**32')
    for key in ('learning_rate', 'head_learning_rate', 'weight_decay'):
        value = training[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'Invalid {key}')
        if value < 0 or (value == 0 and key != 'weight_decay'):
            raise ValueError(f'Invalid {key}')
    return config


def load_config(path: str | Path) -> dict:
    path = Path(path)
    # YAML 1.1 can treat JSON scientific notation such as 1e-05 as a string.
    loader = json.loads if path.suffix == '.json' else yaml.safe_load
    return validate_config(loader(path.read_text()))


def check_comparison(frozen: dict, tuned: dict) -> None:
    validate_config(frozen)
    validate_config(tuned)
    if not frozen['model']['freeze_backbone'] or tuned['model']['freeze_backbone']:
        raise ValueError('Expected frozen and fine-tuned modes in that order')
    if frozen['training'] != tuned['training']:
        raise ValueError('Comparison must share seeds, head/backbone rates, batches and epoch budget')
