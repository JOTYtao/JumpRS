from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.artifacts import artifact_dir
from src.config import load_configs
from src.models.multistep_baselines import (
    ConditionalDiffusionForecaster,
    DeterministicSeqForecaster,
    ITransformerForecaster,
    PatchTSTForecaster,
    QuantileSeqForecaster,
    TimesNetForecaster,
    predict_deterministic,
    predict_diffusion,
    predict_quantile,
)
from src.run_multisite_baselines import baseline_predictions, load_split
from src.run_multisite_experiments import _quantile_pred_from_raw, _sample_pred_from_raw
from src.training.train import _make_jumprs_model, predict_jumprs


MODEL_NAMES = (
    "Persistence",
    "JumpRS",
    "MC Dropout",
    "PatchTST",
    "iTransformer",
    "TimesNet",
    "TimeDiff-style",
    "NsDiff-style",
    "QuantileGRU",
)


def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_site_splits(root: Path, site_id: str):
    path = root / "outputs" / "prepared" / "multisite_splits.npz"
    splits = {name: load_split(path, name) for name in ["train", "validation", "test"]}
    out = {}
    for name, split in splits.items():
        mask = split["site_id"].astype(str) == site_id
        out[name] = {key: value[mask] for key, value in split.items()}
        if not mask.any():
            raise ValueError(f"No {site_id} rows found in the {name} split.")
    return out


def _apply_saved_scaler(split, checkpoint):
    meta = checkpoint.get("metadata", {})
    mean = np.asarray(meta.get("input_scaler_mean"), dtype="float32")
    std = np.asarray(meta.get("input_scaler_std"), dtype="float32")
    if mean.size == 0 or std.size == 0:
        return split
    return {**split, "X_hist": ((split["X_hist"] - mean) / np.maximum(std, 1e-6)).astype("float32")}


def _model_from_checkpoint(model_name, checkpoint, split, data_cfg, model_cfg):
    input_dim = split["X_hist"].shape[-1]
    history_steps = split["X_hist"].shape[1]
    horizon = split["y_x"].shape[1]
    if model_name == "JumpRS":
        model = _make_jumprs_model({"train": split}, data_cfg, model_cfg, "JumpRS")
    elif model_name == "MC Dropout":
        model = DeterministicSeqForecaster(input_dim, horizon, mc_dropout=True)
    elif model_name == "PatchTST":
        model = PatchTSTForecaster(input_dim, horizon, hidden_dim=64, num_layers=1)
    elif model_name == "iTransformer":
        model = ITransformerForecaster(input_dim, history_steps, horizon, hidden_dim=64, num_layers=1)
    elif model_name == "TimesNet":
        model = TimesNetForecaster(input_dim, horizon, hidden_dim=64, num_layers=1, top_k=2)
    elif model_name == "TimeDiff-style":
        model = ConditionalDiffusionForecaster(input_dim, horizon, hidden_dim=128, diffusion_steps=50, kind="timediff")
    elif model_name == "NsDiff-style":
        model = ConditionalDiffusionForecaster(input_dim, horizon, hidden_dim=128, diffusion_steps=50, kind="nsdiff")
    elif model_name == "QuantileGRU":
        model = QuantileSeqForecaster(input_dim, horizon)
    else:
        raise ValueError(f"Unsupported checkpoint model: {model_name}")
    model.load_state_dict(checkpoint["state_dict"])
    return model


def _standardize_samples(pred, sample_count=100):
    if "quantile_power" in pred:
        quantiles = np.asarray(pred["quantiles"], dtype=np.float64)
        targets = np.linspace(0.005, 0.995, sample_count)
        samples = []
        for target in targets:
            hi = int(np.searchsorted(quantiles, target, side="right"))
            hi = min(max(hi, 1), len(quantiles) - 1)
            lo = hi - 1
            alpha = np.clip((target - quantiles[lo]) / max(quantiles[hi] - quantiles[lo], 1e-12), 0.0, 1.0)
            samples.append((1.0 - alpha) * pred["quantile_power"][:, :, lo] + alpha * pred["quantile_power"][:, :, hi])
        return np.stack(samples, axis=0).astype("float32")
    if "samples_power" not in pred:
        return np.repeat(pred["power_mean"][None, :, :], sample_count, axis=0)
    samples = np.sort(np.asarray(pred["samples_power"], dtype="float32"), axis=0)
    if len(samples) == sample_count:
        return samples
    idx = np.linspace(0, len(samples) - 1, sample_count)
    lo = np.floor(idx).astype(int)
    hi = np.ceil(idx).astype(int)
    alpha = (idx - lo).reshape(-1, 1, 1)
    return ((1.0 - alpha) * samples[lo] + alpha * samples[hi]).astype("float32")


def _write_prediction(output_dir, model_name, split, pred, sample_count=100):
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = _standardize_samples(pred, sample_count)
    times = np.asarray(split["time"])
    sample_columns = [f"prediction_sample_{index:03d}" for index in range(1, sample_count + 1)]
    for horizon in range(split["y_power"].shape[1]):
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
        path = output_dir / f"lead_{horizon + 1:02d}_{15 * (horizon + 1):03d}min.csv"
        frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def _predict_one(root, run_name, model_name, splits, data_cfg, model_cfg, output_dir):
    thresholds = data_cfg["ramp"]["thresholds"]
    capacities = {site["site_id"]: float(site["capacity_kw"]) for site in data_cfg["sites"]}
    test = splits["test"]
    if model_name == "Persistence":
        pred = baseline_predictions(test, thresholds, capacities, "Persistence")
        _write_prediction(output_dir, model_name, test, pred)
        return

    ckpt_path = artifact_dir(root, model_name, run_name) / "model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing trained checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    local_test = _apply_saved_scaler(test, checkpoint) if model_name != "JumpRS" else test
    device = _device()
    model = _model_from_checkpoint(model_name, checkpoint, local_test, data_cfg, model_cfg).to(device)

    if model_name == "JumpRS":
        pred = predict_jumprs(model, local_test, data_cfg, model_cfg, device)
    elif model_name == "MC Dropout":
        raw = predict_deterministic(model, local_test, device, mc_samples=30)
        pred = _sample_pred_from_raw(raw, local_test, thresholds)
    elif model_name in {"PatchTST", "iTransformer", "TimesNet"}:
        raw = predict_deterministic(model, local_test, device)
        pred = _quantile_pred_from_raw(raw, local_test, thresholds)
    elif model_name in {"TimeDiff-style", "NsDiff-style"}:
        raw = predict_diffusion(model, local_test, device, n_samples=100)
        pred = _sample_pred_from_raw(raw, local_test, thresholds)
    elif model_name == "QuantileGRU":
        raw = predict_quantile(model, local_test, device)
        pred = _quantile_pred_from_raw(raw, local_test, thresholds)
    else:
        raise ValueError(model_name)
    _write_prediction(output_dir, model_name, test, pred)


def main():
    parser = argparse.ArgumentParser(description="Export predictions from trained JumpRS/benchmark artifacts.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--run-name", default="pvdaq_system_34_rollout_4h")
    parser.add_argument("--site-id", default="pvdaq_system_34")
    parser.add_argument("--model", default="all", choices=("all", *MODEL_NAMES))
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    root = args.root.resolve()
    data_cfg, model_cfg, _ = load_configs(root)
    splits = _load_site_splits(root, args.site_id)
    output_dir = args.output_dir or root / "outputs" / "predictions" / f"{args.site_id}_from_artifacts"
    if output_dir.exists():
        for path in output_dir.glob("lead_*.csv"):
            path.unlink()
    models = MODEL_NAMES if args.model == "all" else (args.model,)
    for model_name in models:
        _predict_one(root, args.run_name, model_name, splits, data_cfg, model_cfg, output_dir)
    print(f"Wrote predictions to {output_dir}")


if __name__ == "__main__":
    main()
