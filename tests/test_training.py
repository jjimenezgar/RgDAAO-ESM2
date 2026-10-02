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


from rgdaao.training import assert_disjoint


def test_undefined_metrics_are_strict_json_null(tmp_path):
    save_evaluation(['A1V', 'C2W'], [0., 1.], [0., 0.], tmp_path)
    text = (tmp_path / 'metrics.json').read_text()
    assert 'NaN' not in text
    assert json.loads(text)['spearman'] is None


def test_duplicate_prediction_ids_fail(tmp_path):
    with pytest.raises(ValueError):
        save_evaluation(['A1V', 'A1V'], [0, 1], [0, 1], tmp_path)
    assert not (tmp_path / 'predictions.csv').exists()


def test_split_overlap_fails():
    frame = pd.DataFrame({'mutation': ['A1V'], 'mutated_sequence': ['VCDE']})
    with pytest.raises(ValueError, match='leakage'):
        assert_disjoint(frame, frame)
