from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.config import load_configs
from src.models.multistep_baselines import sample_crps
from src.run_multisite_baselines import load_split


ROOT = Path(".")
SITE_ID = "pvdaq_system_34"
PREDICTION_DIR = ROOT / "outputs" / "predictions" / "pvdaq_system_34_rollout_4h"
METRICS_DIR = ROOT / "outputs" / "metrics" / "pvdaq_system_34_rollout_4h"
SAMPLE_COLUMNS = [f"prediction_sample_{index:03d}" for index in range(1, 101)]


def _safe_auprc(actual, probability):
    actual = np.asarray(actual, dtype=int)
    if np.unique(actual).size < 2:
        return float("nan")
    return float(average_precision_score(actual, probability))


def predictive_trajectory_mean(samples):
    samples = np.asarray(samples, dtype="float64")
    if samples.ndim != 2:
        raise ValueError(f"Expected predictive samples with shape (member, window); got {samples.shape}.")
    return samples.mean(axis=0)


def _load_test_site():
    split = load_split(ROOT / "outputs" / "prepared" / "multisite_splits.npz", "test")
    mask = split["site_id"].astype(str) == SITE_ID
    site = {key: value[mask] for key, value in split.items()}
    if len(site["site_id"]) == 0:
        raise ValueError(f"No {SITE_ID} rows found in the test split.")
    return site


def _validate_model_frame(frame, model, split, horizon):
    expected_time = pd.Series(split["time"][:, horizon]).astype(str).to_numpy()
    observed_time = frame["target_time"].astype(str).to_numpy()
    if not np.array_equal(observed_time, expected_time):
        raise ValueError(f"{model} lead {horizon + 1}: target times do not match the frozen test split.")
    if not np.allclose(frame["actual_power_kw"], split["y_power"][:, horizon], atol=1e-5):
        raise ValueError(f"{model} lead {horizon + 1}: actual power does not match the frozen test split.")
    numeric = frame[["actual_power_kw", "predicted_power_kw", *SAMPLE_COLUMNS]].to_numpy()
    if not np.isfinite(numeric).all():
        raise ValueError(f"{model} lead {horizon + 1}: predictions contain NaN or Inf.")


def compute_metrics():
    data_cfg, _, _ = load_configs(ROOT)
    split = _load_test_site()
    thresholds = [float(value) for value in data_cfg["ramp"]["thresholds"]]
    horizon_count = int(data_cfg["window"]["forecast_steps"])
    denom = max(float(np.nanmax(split["y_power"]) - np.nanmin(split["y_power"])), 1e-12)

    deterministic_rows = []
    probabilistic_rows = []
    event_rows = []
    previous_samples = {}

    for horizon in range(horizon_count):
        lead_minutes = 15 * (horizon + 1)
        path = PREDICTION_DIR / f"lead_{horizon + 1:02d}_{lead_minutes:03d}min.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing frozen prediction file: {path}")
        frame = pd.read_csv(path)
        models = frame["model"].drop_duplicates().tolist()
        if "Persistence" not in models:
            raise ValueError(f"{path} does not contain the Persistence CRPS reference.")

        lead_crps = {}
        for model in models:
            model_frame = frame[frame["model"] == model].reset_index(drop=True)
            _validate_model_frame(model_frame, model, split, horizon)
            actual = model_frame["actual_power_kw"].to_numpy(dtype="float64")
            samples = model_frame[SAMPLE_COLUMNS].to_numpy(dtype="float64").T
            # Use the same predictive-distribution mean for every model so that
            # deterministic metrics are comparable across probabilistic methods.
            predicted = predictive_trajectory_mean(samples)

            error = predicted - actual
            rmse = float(np.sqrt(np.mean(error**2)))
            deterministic_rows.append(
                {
                    "site_id": SITE_ID,
                    "model": model,
                    "horizon_step": horizon + 1,
                    "lead_minutes": lead_minutes,
                    "n": len(actual),
                    "mae": float(np.mean(np.abs(error))),
                    "rmse": rmse,
                    "nrmse": rmse / denom,
                    "point_prediction_method": "mean_of_100_predictive_trajectories",
                    "nrmse_denominator": "test_power_range",
                    "test_power_range_kw": denom,
                }
            )

            crps = float(np.mean(sample_crps(actual[:, None], samples[:, :, None])))
            lead_crps[model] = crps
            probabilistic_rows.append(
                {
                    "site_id": SITE_ID,
                    "model": model,
                    "horizon_step": horizon + 1,
                    "lead_minutes": lead_minutes,
                    "n": len(actual),
                    "crps": crps,
                    "crps_method": "sample_crps_100_members",
                }
            )

            if horizon == 0:
                previous = np.broadcast_to(split["p_prev"][:, 0], samples.shape)
            else:
                previous = previous_samples[model]
            ramp_fraction = (samples - previous) / split["capacity_kw"][:, horizon][None, :]
            for threshold_index, threshold in enumerate(thresholds):
                for direction, operator, label_key in [
                    ("down", np.less_equal, "y_down"),
                    ("up", np.greater_equal, "y_up"),
                ]:
                    boundary = -threshold if direction == "down" else threshold
                    probability = operator(ramp_fraction, boundary).mean(axis=0)
                    actual_event = split[label_key][:, horizon, threshold_index].astype(int)
                    event_rows.append(
                        {
                            "site_id": SITE_ID,
                            "model": model,
                            "horizon_step": horizon + 1,
                            "lead_minutes": lead_minutes,
                            "threshold": threshold,
                            "direction": direction,
                            "n": len(actual_event),
                            "events": int(actual_event.sum()),
                            "event_rate": float(actual_event.mean()),
                            "brier_score": float(np.mean((probability - actual_event) ** 2)),
                            "auprc": _safe_auprc(actual_event, probability),
                            "probability_method": "recursive_sample_frequency_100_members",
                        }
                    )
            previous_samples[model] = samples

        persistence_crps = lead_crps["Persistence"]
        for row in probabilistic_rows[-len(models) :]:
            row["persistence_crps"] = persistence_crps
            row["crpss"] = 1.0 - row["crps"] / max(persistence_crps, 1e-12)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    deterministic = pd.DataFrame(deterministic_rows)
    probabilistic = pd.DataFrame(probabilistic_rows)
    events = pd.DataFrame(event_rows)
    deterministic.to_csv(METRICS_DIR / "deterministic_metrics_by_leading_time.csv", index=False)
    probabilistic.to_csv(METRICS_DIR / "probabilistic_metrics_by_leading_time.csv", index=False)
    events.to_csv(METRICS_DIR / "event_metrics_by_leading_time.csv", index=False)
    print(f"Wrote PVDAQ 34 lead-resolved metrics to {METRICS_DIR}")


if __name__ == "__main__":
    compute_metrics()
