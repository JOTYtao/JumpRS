from pathlib import Path

import numpy as np
import pandas as pd

from src.artifacts import save_model_artifact, save_spec_artifact
from src.config import load_configs
from src.run_multisite_baselines import (
    baseline_predictions,
    event_rows,
    event_rows_by_horizon,
    load_split,
    trajectory_rows,
    trajectory_rows_by_horizon,
)
from src.training.train import predict_jumprs, train_jumprs
from src.models.multistep_baselines import (
    event_probs_from_gaussian_power,
    predict_deterministic,
    predict_diffusion,
    predict_quantile,
    sample_crps,
    train_deterministic,
    train_diffusion,
    train_quantile,
    train_sota_deterministic,
)


def _split_dict(root):
    path = root / "outputs" / "prepared" / "multisite_splits.npz"
    return {name: load_split(path, name) for name in ["train", "validation", "test"]}


def _capacity_by_site(data_cfg):
    return {site["site_id"]: float(site["capacity_kw"]) for site in data_cfg["sites"]}


def _power_pred_from_x(pred, split):
    out = dict(pred)
    out["power_mean"] = pred["mean_x"] * split["p_cs_next"]
    return out


def _quantile_pred(model, splits, device, thresholds):
    raw = predict_quantile(model, splits["test"], device)
    return _quantile_pred_from_raw(raw, splits["test"], thresholds)


def _quantile_pred_from_raw(raw, split, thresholds):
    qs = np.asarray(raw["quantiles"], dtype="float32")
    qx = raw["quantile_x"]
    median = raw["mean_x"]
    # Use q10-q90 as a Gaussian spread approximation for event probabilities.
    q10 = qx[:, :, int(np.argmin(np.abs(qs - 0.1)))]
    q90 = qx[:, :, int(np.argmin(np.abs(qs - 0.9)))]
    std_x = np.maximum((q90 - q10) / (2.0 * 1.2815515655446004), 1e-4)
    down, up = event_probs_from_gaussian_power(median, std_x, split, thresholds)
    pred = _power_pred_from_x({"mean_x": median, "down_prob": down, "up_prob": up}, split)
    pred["quantiles"] = tuple(float(q) for q in qs)
    pred["quantile_power"] = qx * split["p_cs_next"][:, :, None]
    return pred


def _mc_dropout_pred(model, splits, device, thresholds):
    raw = predict_deterministic(model, splits["test"], device, mc_samples=30)
    return _sample_pred_from_raw(raw, splits["test"], thresholds)


def _sample_pred_from_raw(raw, split, thresholds):
    samples_x = raw["samples_x"]
    samples_power = samples_x * split["p_cs_next"][None, :, :]
    mean_x = raw["mean_x"]
    # Avoid peeking at future ground-truth "previous" power for multi-step ramps:
    # use the last observed power for step 1 and the sample's prior-step prediction thereafter.
    prev = np.zeros_like(samples_power, dtype="float32")
    prev[:, :, 0] = split["p_prev"][:, 0]
    if samples_power.shape[-1] > 1:
        prev[:, :, 1:] = samples_power[:, :, :-1]
    ramp = (samples_power - prev) / split["capacity_kw"][None, :, :]
    down = np.stack([(ramp <= -float(g)).mean(axis=0) for g in thresholds], axis=-1)
    up = np.stack([(ramp >= float(g)).mean(axis=0) for g in thresholds], axis=-1)
    pred = _power_pred_from_x({"mean_x": mean_x, "down_prob": down, "up_prob": up}, split)
    pred["samples_power"] = samples_power
    return pred


def crps_array(split, pred):
    y = split["y_power"]
    crps = None
    method = ""
    if "quantile_power" in pred:
        qs = np.asarray(pred["quantiles"])
        q = pred["quantile_power"]
        err = y[:, :, None] - q
        pinball = np.maximum(qs.reshape(1, 1, -1) * err, (qs.reshape(1, 1, -1) - 1.0) * err)
        crps = 2.0 * pinball.mean(axis=-1)
        method = "quantile_pinball_approx"
    elif "samples_power" in pred:
        crps = sample_crps(y, pred["samples_power"])
        method = "sample_crps"
    elif "power_mean" in pred:
        # For a deterministic forecast, CRPS reduces to absolute error.
        crps = np.abs(y - pred["power_mean"])
        method = "deterministic_crps_equals_absolute_error"
    return crps, method


def crps_rows(model_name, split, pred):
    rows = []
    y = split["y_power"]
    crps, method = crps_array(split, pred)
    if crps is None:
        return rows
    groups = {
        "all": np.ones(y.shape[1], dtype=bool),
        "0-1h": np.arange(y.shape[1]) < 4,
        "1-3h": (np.arange(y.shape[1]) >= 4) & (np.arange(y.shape[1]) < 12),
        "3-6h": np.arange(y.shape[1]) >= 12,
    }
    for group_name, hmask in groups.items():
        rows.append({"model": model_name, "horizon_group": group_name, "crps": float(np.mean(crps[:, hmask])), "method": method})
    return rows


def crps_rows_by_horizon(model_name, split, pred):
    crps, method = crps_array(split, pred)
    if crps is None:
        return []
    return [
        {
            "model": model_name,
            "horizon_step": horizon + 1,
            "lead_minutes": 15 * (horizon + 1),
            "crps": float(np.mean(crps[:, horizon])),
            "method": method,
        }
        for horizon in range(crps.shape[1])
    ]


def write_test_sample_predictions(path, split, preds, thresholds):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    times = np.asarray(split["time"])
    origin_times = times[:, 0] if times.ndim == 2 else times
    target_times = times if times.ndim == 2 else np.repeat(times[:, None], split["y_power"].shape[1], axis=1)
    site_ids = split["site_id"].astype(str)
    horizon = split["y_power"].shape[1]
    first = True
    base = pd.DataFrame({
        "sample_id": np.repeat(np.arange(len(site_ids)), horizon),
        "site_id": np.repeat(site_ids, horizon),
        "forecast_origin_time": np.repeat(pd.Series(origin_times).astype(str).to_numpy(), horizon),
        "target_time": pd.Series(target_times.reshape(-1)).astype(str).to_numpy(),
        "horizon_step": np.tile(np.arange(1, horizon + 1), len(site_ids)),
        "lead_minutes": np.tile(15 * np.arange(1, horizon + 1), len(site_ids)),
        "actual_power_kw": split["y_power"].reshape(-1),
        "actual_x": split["y_x"].reshape(-1),
        "p_prev_kw": split["p_prev"].reshape(-1),
        "p_cs_next_kw": split["p_cs_next"].reshape(-1),
    })
    for i, gamma in enumerate(thresholds):
        suffix = f"{float(gamma):.2f}"
        base[f"actual_down_g{suffix}"] = split["y_down"][:, :, i].reshape(-1).astype(int)
        base[f"actual_up_g{suffix}"] = split["y_up"][:, :, i].reshape(-1).astype(int)
    for model_name, pred in preds.items():
        out = base.copy()
        out.insert(0, "model", model_name)
        sample_crps, crps_method = crps_array(split, pred)
        if sample_crps is not None:
            out["crps_kw"] = sample_crps.reshape(-1)
            out["crps_method"] = crps_method
        else:
            out["crps_kw"] = np.nan
            out["crps_method"] = "not_defined"
        power = pred["power_mean"]
        out["predicted_power_kw"] = power.reshape(-1)
        out["power_error_kw"] = (power - split["y_power"]).reshape(-1)
        mean_x = pred.get("mean_x")
        if mean_x is None:
            mean_x = power / np.maximum(split["p_cs_next"], 1e-6)
        out["predicted_x"] = mean_x.reshape(-1)
        for i, gamma in enumerate(thresholds):
            suffix = f"{float(gamma):.2f}"
            out[f"pred_down_prob_g{suffix}"] = pred["down_prob"][:, :, i].reshape(-1)
            out[f"pred_up_prob_g{suffix}"] = pred["up_prob"][:, :, i].reshape(-1)
        out.to_csv(path, mode="w" if first else "a", index=False, header=first)
        first = False


def run():
    root = Path(".")
    data_cfg, model_cfg, train_cfg = load_configs(root)
    splits = _split_dict(root)
    thresholds = data_cfg["ramp"]["thresholds"]
    metrics_dir = root / "outputs" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    preds = {}
    histories = []
    capacity_by_site = _capacity_by_site(data_cfg)
    artifact_run = "multisite_current"
    common_meta = {
        "dataset": data_cfg["dataset"]["name"],
        "sites": [site["site_id"] for site in data_cfg["sites"]],
        "history_steps": int(data_cfg["window"]["history_steps"]),
        "forecast_steps": int(data_cfg["window"]["forecast_steps"]),
        "resolution": data_cfg["preprocessing"]["resample_rule"],
        "data_config": data_cfg,
        "model_config": model_cfg,
        "train_config": train_cfg,
    }
    for model_name in ["Persistence"]:
        preds[model_name] = baseline_predictions(splits["test"], thresholds, capacity_by_site, model_name)
        save_spec_artifact(root, model_name, artifact_run, {**common_meta, "kind": "non_parametric"})

    proposed_encoder = model_cfg["model"].get("encoder_type", "JumpRS")
    proposed_model, hist, proposed_device = train_jumprs(
        splits,
        data_cfg,
        model_cfg,
        train_cfg,
        encoder_type=proposed_encoder,
        model_name="JumpRS",
    )
    histories.extend(hist)
    save_model_artifact(
        proposed_model,
        root,
        "JumpRS",
        artifact_run,
        {**common_meta, "kind": "proposed", "device": str(proposed_device)},
        hist,
    )
    preds["JumpRS"] = predict_jumprs(
        proposed_model,
        splits["test"],
        data_cfg,
        model_cfg,
        proposed_device,
    )

    for name, mc in [("MC Dropout", True)]:
        det_model, local_splits, hist, det_device = train_deterministic(name, splits, train_cfg, mc_dropout=mc)
        histories.extend(hist)
        save_model_artifact(
            det_model,
            root,
            name,
            artifact_run,
            {
                **common_meta,
                "kind": "benchmark",
                "device": str(det_device),
                "input_scaler_mean": local_splits["train"].get("_x_mean"),
                "input_scaler_std": local_splits["train"].get("_x_std"),
            },
            hist,
        )
        preds[name] = _mc_dropout_pred(det_model, local_splits, det_device, thresholds)

    for name in ["PatchTST", "iTransformer", "TimesNet"]:
        sota_model, local_splits, hist, sota_device = train_sota_deterministic(name, splits, train_cfg)
        histories.extend(hist)
        save_model_artifact(
            sota_model,
            root,
            name,
            artifact_run,
            {
                **common_meta,
                "kind": "benchmark",
                "device": str(sota_device),
                "input_scaler_mean": local_splits["train"].get("_x_mean"),
                "input_scaler_std": local_splits["train"].get("_x_std"),
            },
            hist,
        )
        raw = predict_deterministic(sota_model, local_splits["test"], sota_device)
        preds[name] = _quantile_pred_from_raw(raw, local_splits["test"], thresholds)

    for name in ["TimeDiff-style", "NsDiff-style"]:
        diffusion_model, local_splits, hist, diffusion_device = train_diffusion(name, splits, train_cfg)
        histories.extend(hist)
        save_model_artifact(
            diffusion_model,
            root,
            name,
            artifact_run,
            {
                **common_meta,
                "kind": "benchmark",
                "device": str(diffusion_device),
                "input_scaler_mean": local_splits["train"].get("_x_mean"),
                "input_scaler_std": local_splits["train"].get("_x_std"),
            },
            hist,
        )
        raw = predict_diffusion(diffusion_model, local_splits["test"], diffusion_device, n_samples=100)
        preds[name] = _sample_pred_from_raw(raw, local_splits["test"], thresholds)

    q_model, q_splits, hist, q_device = train_quantile(splits, train_cfg)
    histories.extend(hist)
    save_model_artifact(
        q_model,
        root,
        "QuantileGRU",
        artifact_run,
        {
            **common_meta,
            "kind": "benchmark",
            "device": str(q_device),
            "input_scaler_mean": q_splits["train"].get("_x_mean"),
            "input_scaler_std": q_splits["train"].get("_x_std"),
        },
        hist,
    )
    preds["QuantileGRU"] = _quantile_pred(q_model, q_splits, q_device, thresholds)

    event_all, event_horizon, traj_all, traj_horizon, crps_all, crps_horizon = [], [], [], [], [], []
    for model_name, pred in preds.items():
        event_all.extend(event_rows(model_name, splits["test"], pred, thresholds))
        event_horizon.extend(event_rows_by_horizon(model_name, splits["test"], pred, thresholds))
        traj_all.extend(trajectory_rows(model_name, splits["test"], pred))
        traj_horizon.extend(trajectory_rows_by_horizon(model_name, splits["test"], pred))
        crps_all.extend(crps_rows(model_name, splits["test"], pred))
        crps_horizon.extend(crps_rows_by_horizon(model_name, splits["test"], pred))

    event_all_cols = ["model", "site_id", "horizon_group", "threshold", "direction", "brier", "auprc", "n", "events"]
    event_horizon_cols = ["model", "horizon_step", "lead_minutes", "threshold", "direction", "brier", "auprc", "n", "events"]
    pd.DataFrame(event_all)[event_all_cols].to_csv(metrics_dir / "main_event_results_multisite.csv", index=False)
    pd.DataFrame(event_horizon)[event_horizon_cols].to_csv(metrics_dir / "main_event_results_by_horizon.csv", index=False)
    pd.DataFrame(traj_all).to_csv(metrics_dir / "trajectory_results_multisite.csv", index=False)
    pd.DataFrame(traj_horizon).to_csv(metrics_dir / "trajectory_results_by_horizon.csv", index=False)
    crps_all_df = pd.DataFrame(crps_all)
    crps_horizon_df = pd.DataFrame(crps_horizon)
    crps_all_df.to_csv(metrics_dir / "crps_results_multisite.csv", index=False)
    crps_horizon_df.to_csv(metrics_dir / "crps_results_by_horizon.csv", index=False)
    if not crps_all_df.empty and "Persistence" in set(crps_all_df["model"]):
        ref = crps_all_df[crps_all_df["model"] == "Persistence"][["horizon_group", "crps"]].rename(columns={"crps": "persistence_crps"})
        crps_skill = crps_all_df.merge(ref, on="horizon_group", how="left")
        crps_skill["crpss_vs_persistence"] = 1.0 - crps_skill["crps"] / crps_skill["persistence_crps"].clip(lower=1e-12)
        crps_skill.to_csv(metrics_dir / "crps_skill_results_multisite.csv", index=False)
    if not crps_horizon_df.empty and "Persistence" in set(crps_horizon_df["model"]):
        ref_h = crps_horizon_df[crps_horizon_df["model"] == "Persistence"][["horizon_step", "crps"]].rename(columns={"crps": "persistence_crps"})
        crps_skill_h = crps_horizon_df.merge(ref_h, on="horizon_step", how="left")
        crps_skill_h["crpss_vs_persistence"] = 1.0 - crps_skill_h["crps"] / crps_skill_h["persistence_crps"].clip(lower=1e-12)
        crps_skill_h.to_csv(metrics_dir / "crps_skill_results_by_horizon.csv", index=False)
    pd.DataFrame(histories).to_csv(metrics_dir / "training_history_multisite.csv", index=False)
    write_test_sample_predictions(root / "outputs" / "predictions" / "raw_test_sample_predictions_multisite.csv", splits["test"], preds, thresholds)
    print("Completed multisite JumpRS/probabilistic baseline experiments.")


if __name__ == "__main__":
    run()
