import numpy as np

from src.evaluation.compute_pvdaq34_lead_metrics import (
    _safe_auprc,
    predictive_trajectory_mean,
)


def test_safe_auprc_is_nan_for_single_class():
    assert np.isnan(_safe_auprc([0, 0, 0], [0.1, 0.2, 0.3]))


def test_safe_auprc_is_finite_for_two_classes():
    assert np.isfinite(_safe_auprc([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8]))


def test_deterministic_point_prediction_is_predictive_trajectory_mean():
    samples = np.array([[1.0, 4.0], [3.0, 8.0], [5.0, 12.0]])
    np.testing.assert_allclose(predictive_trajectory_mean(samples), [3.0, 8.0])
