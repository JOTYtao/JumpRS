from pathlib import Path
import argparse
import json
import shutil

import numpy as np
import pandas as pd

from src.artifacts import save_model_artifact, save_spec_artifact
from src.config import load_configs
from src.models.multistep_baselines import (
    predict_deterministic,
    predict_diffusion,
    predict_quantile,
    train_deterministic,
    train_diffusion,
    train_quantile,
    train_sota_deterministic,
)
from src.run_multisite_baselines import baseline_predictions, load_split
from src.run_multisite_experiments import _quantile_pred_from_raw, _sample_pred_from_raw
from src.training.train import predict_jumprs, train_jumprs


ROOT = Path(".")
SITE_ID = "pvdaq_system_34"
SAMPLE_COUNT = 100
OUT = ROOT / "outputs" / "predictions" / "pvdaq_system_34_rollout_4h"
LOG_OUT = ROOT / "outputs" / "logs" / "pvdaq_system_34_rollout_4h"
ARTIFACT_RUN = "pvdaq_system_34_rollout_4h"
MODEL_ORDER = [
    "Persistence",
    "JumpRS",
    "MC Dropout",
    "PatchTST",
    "iTransformer",
    "TimesNet",
    "TimeDiff-style",
    "NsDiff-style",
    "QuantileGRU",
]


def _load_site_splits():
    path = ROOT / "outputs" / "prepared" / "multisite_splits.npz"
    splits = {name: load_split(path, name) for name in ["train", "validation", "test"]}
    out = {}
    for name, split in splits.items():
        mask = split["site_id"].astype(str) == SITE_ID
        out[name] = {key: value[mask] for key, value in split.items()}
        if not mask.any():
            raise ValueError(f"No {SITE_ID} rows found in the {name} split.")
    return out


def _quantiles_to_samples(quantile_power, quantiles):
    quantiles = np.asarray(quantiles, dtype=np.float64)
    targets = np.linspace(0.005, 0.995, SAMPLE_COUNT)
    samples = []
    for target in targets:
        hi = int(np.searchsorted(quantiles, target, side="right"))
        hi = min(max(hi, 1), len(quantiles) - 1)
        lo = hi - 1
        alpha = np.clip(
            (target - quantiles[lo]) / max(quantiles[hi] - quantiles[lo], 1e-12),
            0.0,
            1.0,
        )
        samples.append(
            (1.0 - alpha) * quantile_power[:, :, lo]
            + alpha * quantile_power[:, :, hi]
        )
    return np.stack(samples, axis=0).astype("float32")


def _standardize_samples(pred):
    if "quantile_power" in pred:
        return _quantiles_to_samples(pred["quantile_power"], pred["quantiles"])
    if "samples_power" not in pred:
        return np.repeat(pred["power_mean"][None, :, :], SAMPLE_COUNT, axis=0)
    samples = np.sort(np.asarray(pred["samples_power"], dtype="float32"), axis=0)
    if len(samples) == SAMPLE_COUNT:
        return samples
    idx = np.linspace(0, len(samples) - 1, SAMPLE_COUNT)
    lo = np.floor(idx).astype(int)
    hi = np.ceil(idx).astype(int)
    alpha = (idx - lo).reshape(-1, 1, 1)
    return ((1.0 - alpha) * samples[lo] + alpha * samples[hi]).astype("float32")


def _write_prediction(model_name, split, pred):
    samples = _standardize_samples(pred)
    times = np.asarray(split["time"])
    if times.ndim != 2 or times.shape[1] != 16:
        raise ValueError(f"Expected PVDAQ 34 test timestamps with shape (N,16); got {times.shape}.")
    sample_columns = [f"prediction_sample_{index:03d}" for index in range(1, SAMPLE_COUNT + 1)]
    for horizon in range(16):
        base = pd.DataFrame(
            {
                "model": model_name,
                "target_time": pd.Series(times[:, horizon]).astype(str),
                "actual_power_kw": split["y_power"][:, horizon],
                "predicted_power_kw": pred["power_mean"][:, horizon],
            }
        )
        sample_frame = pd.DataFrame(samples[:, :, horizon].T, columns=sample_columns)
        frame = pd.concat([base, sample_frame], axis=1)
        path = OUT / f"lead_{horizon + 1:02d}_{15 * (horizon + 1):03d}min.csv"
        frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def _selected(models):
    if not models or "all" in models:
        return set(MODEL_ORDER)
    unknown = sorted(set(models) - set(MODEL_ORDER))
    if unknown:
        raise ValueError(f"Unknown model names: {unknown}. Valid names are: {MODEL_ORDER}")
    return set(models)


def run(models=None):
    selected = _selected(models)
    data_cfg, model_cfg, train_cfg = load_configs(ROOT)
    if int(data_cfg["window"]["forecast_steps"]) != 16:
        raise ValueError("PVDAQ 34 rollout experiment requires forecast_steps=16.")
    if model_cfg["model"].get("conditioning_path_mode") != "mean":
        raise ValueError("PVDAQ 34 rollout experiment requires recursive mean conditioning.")
    if not bool(model_cfg["model"].get("use_mc_samples_for_intervals", False)):
        raise ValueError("PVDAQ 34 rollout experiment requires recursive MC trajectory samples.")

    splits = _load_site_splits()
    if OUT.exists():
        shutil.rmtree(OUT)
    if LOG_OUT.exists():
        shutil.rmtree(LOG_OUT)
    OUT.mkdir(parents=True)
    LOG_OUT.mkdir(parents=True)
    histories = []
    thresholds = data_cfg["ramp"]["thresholds"]
    capacities = {site["site_id"]: float(site["capacity_kw"]) for site in data_cfg["sites"]}
    common_meta = {
        "dataset": data_cfg["dataset"]["name"],
        "site_id": SITE_ID,
        "history_steps": int(data_cfg["window"]["history_steps"]),
        "forecast_steps": int(data_cfg["window"]["forecast_steps"]),
        "resolution": data_cfg["preprocessing"]["resample_rule"],
        "data_config": data_cfg,
        "model_config": model_cfg,
        "train_config": train_cfg,
    }

    if "Persistence" in selected:
        persistence = baseline_predictions(splits["test"], thresholds, capacities, "Persistence")
        save_spec_artifact(ROOT, "Persistence", ARTIFACT_RUN, {**common_meta, "kind": "non_parametric"})
        _write_prediction("Persistence", splits["test"], persistence)

    if "JumpRS" in selected:
        model, history, device = train_jumprs(
            splits, data_cfg, model_cfg, train_cfg, encoder_type="JumpRS", model_name="JumpRS-PVDAQ34"
        )
        histories.extend(history)
        save_model_artifact(
            model,
            ROOT,
            "JumpRS",
            ARTIFACT_RUN,
            {**common_meta, "kind": "proposed", "device": str(device)},
            history,
        )
        _write_prediction("JumpRS", splits["test"], predict_jumprs(model, splits["test"], data_cfg, model_cfg, device))

    if "MC Dropout" in selected:
        model, local, history, device = train_deterministic("MC Dropout", splits, train_cfg, mc_dropout=True)
        histories.extend(history)
        save_model_artifact(
            model,
            ROOT,
            name,
            ARTIFACT_RUN,
            {
                **common_meta,
                "kind": "benchmark",
                "device": str(device),
                "input_scaler_mean": local["train"].get("_x_mean"),
                "input_scaler_std": local["train"].get("_x_std"),
            },
            history,
        )
        raw = predict_deterministic(model, local["test"], device, mc_samples=30)
        _write_prediction("MC Dropout", local["test"], _sample_pred_from_raw(raw, local["test"], thresholds))

    for name in ["PatchTST", "iTransformer", "TimesNet"]:
        if name not in selected:
            continue
        model, local, history, device = train_sota_deterministic(name, splits, train_cfg)
        histories.extend(history)
        save_model_artifact(
            model,
            ROOT,
            name,
            ARTIFACT_RUN,
            {
                **common_meta,
                "kind": "benchmark",
                "device": str(device),
                "input_scaler_mean": local["train"].get("_x_mean"),
                "input_scaler_std": local["train"].get("_x_std"),
            },
            history,
        )
        raw = predict_deterministic(model, local["test"], device)
        _write_prediction(name, local["test"], _quantile_pred_from_raw(raw, local["test"], thresholds))

    for name in ["TimeDiff-style", "NsDiff-style"]:
        if name not in selected:
            continue
        model, local, history, device = train_diffusion(name, splits, train_cfg)
        histories.extend(history)
        save_model_artifact(
            model,
            ROOT,
            name,
            ARTIFACT_RUN,
            {
                **common_meta,
                "kind": "benchmark",
                "device": str(device),
                "input_scaler_mean": local["train"].get("_x_mean"),
                "input_scaler_std": local["train"].get("_x_std"),
            },
            history,
        )
        raw = predict_diffusion(model, local["test"], device, n_samples=SAMPLE_COUNT)
        _write_prediction(name, local["test"], _sample_pred_from_raw(raw, local["test"], thresholds))

    if "QuantileGRU" in selected:
        model, local, history, device = train_quantile(splits, train_cfg)
        histories.extend(history)
        save_model_artifact(
            model,
            ROOT,
            "QuantileGRU",
            ARTIFACT_RUN,
            {
                **common_meta,
                "kind": "benchmark",
                "device": str(device),
                "input_scaler_mean": local["train"].get("_x_mean"),
                "input_scaler_std": local["train"].get("_x_std"),
            },
            history,
        )
        raw = predict_quantile(model, local["test"], device)
        _write_prediction("QuantileGRU", local["test"], _quantile_pred_from_raw(raw, local["test"], thresholds))

    pd.DataFrame(histories).to_csv(LOG_OUT / "training_history.csv", index=False)
    manifest = {
        "site_id": SITE_ID,
        "forecast_steps": 16,
        "resolution_minutes": 15,
        "forecast_hours": 4,
        "prediction_samples_per_model": SAMPLE_COUNT,
        "models": sorted(selected),
    }
    (LOG_OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote PVDAQ 34 rollout predictions to {OUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and export PVDAQ 34 rollout predictions.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="Model names to run, or all. Use quotes for names with spaces, e.g. 'MC Dropout'.",
    )
    args = parser.parse_args()
    run(args.models)
