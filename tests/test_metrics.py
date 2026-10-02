import pytest
from rgdaao.metrics import regression_metrics


def test_perfect_predictions():
    y = [0.1, 0.2, 0.4, 0.9]
    m = regression_metrics(y, y)
    assert m["pearson"] == pytest.approx(1.0)
    assert m["spearman"] == pytest.approx(1.0)
    assert m["mae"] == pytest.approx(0.0)
    assert m["rmse"] == pytest.approx(0.0)
