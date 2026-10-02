import pytest
from rgdaao.metrics import regression_metrics


def test_perfect_predictions():
    y = [0.1, 0.2, 0.4, 0.9]
    m = regression_metrics(y, y)
    assert m["pearson"] == pytest.approx(1.0)
    assert m["spearman"] == pytest.approx(1.0)
    assert m["mae"] == pytest.approx(0.0)
    assert m["rmse"] == pytest.approx(0.0)


@pytest.mark.parametrize('truth,pred', [([1], [1]), ([1, 2], [1]), ([1, float('inf')], [1, 2]), ([1, 2], [0, float('nan')])])
def test_invalid_metrics_fail(truth, pred):
    with pytest.raises(ValueError):
        regression_metrics(truth, pred)


def test_constant_predictions_undefined_correlation():
    m = regression_metrics([0, 1, 2], [1, 1, 1])
    assert m['spearman'] is None and m['pearson'] is None
    assert m['mae'] == pytest.approx(2/3)
    assert m['rmse'] == pytest.approx((2/3)**0.5)


def test_known_nonperfect_predictions():
    m = regression_metrics([1, 2, 3], [3, 2, 1])
    assert m['spearman'] == pytest.approx(-1)
    assert m['mae'] == pytest.approx(4/3)
