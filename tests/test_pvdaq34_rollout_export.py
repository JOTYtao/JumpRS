import numpy as np
import pandas as pd

from src.run_pvdaq34_rollout_experiments import (
    SAMPLE_COUNT,
    _quantiles_to_samples,
    _write_prediction,
)


def test_quantile_predictions_are_standardized_to_100_samples():
    quantiles = (0.1, 0.5, 0.9)
    values = np.array([[[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]], dtype="float32")
    samples = _quantiles_to_samples(values, quantiles)
    assert samples.shape == (100, 1, 2)
    assert np.all(np.diff(samples[:, 0, 0]) >= 0)


def test_lead_csv_contains_only_requested_prediction_fields(tmp_path, monkeypatch):
    monkeypatch.setattr("src.run_pvdaq34_rollout_experiments.OUT", tmp_path)
    split = {
        "time": np.array([["2024-01-01T00:15:00"] * 16]),
        "y_power": np.ones((1, 16), dtype="float32"),
    }
    pred = {
        "power_mean": np.full((1, 16), 2.0, dtype="float32"),
        "samples_power": np.full((3, 1, 16), 2.0, dtype="float32"),
    }
    _write_prediction("Example", split, pred)
    frame = pd.read_csv(tmp_path / "lead_01_015min.csv")
    assert list(frame.columns[:4]) == [
        "model",
        "target_time",
        "actual_power_kw",
        "predicted_power_kw",
    ]
    assert len(frame.columns) == 4 + SAMPLE_COUNT
