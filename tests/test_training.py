import json
import pandas as pd
import pytest
from rgdaao.training import save_evaluation


def test_save_evaluation(tmp_path):
    metrics = save_evaluation(
        ["A1V", "C2W"],
        [0.0, 1.0],
        [0.0, 1.0],
        tmp_path,
    )
    assert metrics["rmse"] == 0.0
    assert (tmp_path / "predictions.csv").exists()
    saved = json.loads((tmp_path / "metrics.json").read_text())
    assert saved["spearman"] == pytest.approx(1.0)
