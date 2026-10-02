from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import regression_metrics


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def save_evaluation(
    mutations: list[str],
    y_true,
    y_pred,
    output_dir: str | Path,
) -> dict[str, float]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics = regression_metrics(y_true, y_pred)
    pd.DataFrame({
        "mutation": mutations,
        "experimental_activity": y_true,
        "predicted_activity": y_pred,
    }).to_csv(output / "predictions.csv", index=False)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics
