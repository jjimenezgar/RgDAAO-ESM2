from __future__ import annotations
import hashlib
import importlib.metadata
import json
import os
import random
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from .metrics import regression_metrics


def set_seed(seed: int) -> None:
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    except ImportError:
        pass


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment():
    packages = {n: importlib.metadata.version(n) for n in
                ('rgdaao-esm2', 'numpy', 'pandas', 'scipy', 'torch', 'transformers', 'accelerate')}
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    import torch
    return {'packages': packages, 'git_commit': commit, 'git_dirty': dirty,
            'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
            'cuda': torch.version.cuda}


def assert_disjoint(*frames):
    for col in ('mutation', 'mutated_sequence'):
        seen = set()
        for frame in frames:
            values = set(frame[col])
            if seen & values:
                raise ValueError(f'Split leakage: overlapping {col}')
            seen.update(values)


def save_evaluation(mutations, y_true, y_pred, output_dir):
    if len(mutations) != len(y_true) or len(set(mutations)) != len(mutations):
        raise ValueError('Evaluation requires one unique mutation identifier per prediction')
    metrics = regression_metrics(y_true, y_pred)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'mutation': mutations, 'experimental_activity': y_true,
                  'predicted_activity': y_pred}).to_csv(output / 'predictions.csv', index=False)
    write_json(output / 'metrics.json', metrics)
    return metrics
