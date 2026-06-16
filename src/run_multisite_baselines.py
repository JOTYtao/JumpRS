from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import load_configs


def load_split(npz_path, split_name):
    data = np.load(npz_path, allow_pickle=True)
    prefix = f"{split_name}__"
    return {k[len(prefix):]: data[k] for k in data.files if k.startswith(prefix)}


def safe_score(fn, y, p):
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(fn(y, p))


def ece_score(y, p, bins=10):
    y = np.asarray(y).astype(float)
    p = np.asarray(p).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y)
    if total == 0:
        return float("nan")
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if not mask.any():
            continue
        ece += mask.mean() * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return float(ece)


def baseline_predictions(split, thresholds, capacity_kw_by_site, kind):
    hist = split["X_hist"]
    sites = split["site_id"].astype(str)
    p_cs_next = split["p_cs_next"]
    p_prev = split["p_prev"]
    if kind == "Persistence":
        pred_x = np.repeat(hist[:, -1:, 0], p_cs_next.shape[1], axis=1)
    else:
        raise ValueError(kind)
    power_mean = pred_x * p_cs_next
    capacity = np.asarray([capacity_kw_by_site[s] for s in sites], dtype="float32").reshape(-1, 1)
    # Multi-step ramp events are defined between consecutive future steps. For forecasting,
    # the "previous" power is only known for the first step; subsequent steps must use the
    # model's own prior-step prediction (no peeking at ground truth future power).
    if p_prev.ndim == 2:
        prev0 = p_prev[:, 0]
    else:
        prev0 = p_prev
    prev_power = np.zeros_like(power_mean, dtype="float32")
    prev_power[:, 0] = prev0.astype("float32")
    if power_mean.shape[1] > 1:
        prev_power[:, 1:] = power_mean[:, :-1]
    ramp = (power_mean - prev_power) / capacity
    down = np.stack([(ramp <= -float(g)).astype(float) for g in thresholds], axis=-1)
    up = np.stack([(ramp >= float(g)).astype(float) for g in thresholds], axis=-1)
    return {"power_mean": power_mean, "down_prob": down, "up_prob": up}


def event_rows(model, split, pred, thresholds):
    rows = []
    groups = [("all", np.ones(split["y_down"].shape[1], dtype=bool))]
    groups.extend([
        ("0-1h", np.arange(split["y_down"].shape[1]) < 4),
        ("1-3h", (np.arange(split["y_down"].shape[1]) >= 4) & (np.arange(split["y_down"].shape[1]) < 12)),
        ("3-6h", np.arange(split["y_down"].shape[1]) >= 12),
    ])
    for site_id in ["ALL", *sorted(set(split["site_id"].astype(str)))]:
        site_mask = np.ones(len(split["site_id"]), dtype=bool) if site_id == "ALL" else split["site_id"].astype(str) == site_id
        for group_name, hmask in groups:
            for i, gamma in enumerate(thresholds):
                for direction, y_key, p_key in [("down", "y_down", "down_prob"), ("up", "y_up", "up_prob")]:
                    y = split[y_key][site_mask][:, hmask, i].reshape(-1).astype(int)
                    p = pred[p_key][site_mask][:, hmask, i].reshape(-1)
                    hard = (p >= 0.5).astype(int)
                    rows.append({
                        "model": model,
                        "site_id": site_id,
                        "horizon_group": group_name,
                        "threshold": gamma,
                        "direction": direction,
                        "brier": float(brier_score_loss(y, p)),
                        "bss_vs_persistence": np.nan,
                        "auprc": safe_score(average_precision_score, y, p),
                        "auroc": safe_score(roc_auc_score, y, p),
                        "f1": float(f1_score(y, hard, zero_division=0)),
                        "precision": float(precision_score(y, hard, zero_division=0)),
                        "recall": float(recall_score(y, hard, zero_division=0)),
                        "ece": ece_score(y, p),
                        "n": int(len(y)),
                        "events": int(y.sum()),
                    })
    return rows


def event_rows_by_horizon(model, split, pred, thresholds):
    rows = []
    for h in range(split["y_down"].shape[1]):
        for i, gamma in enumerate(thresholds):
            for direction, y_key, p_key in [("down", "y_down", "down_prob"), ("up", "y_up", "up_prob")]:
                y = split[y_key][:, h, i].reshape(-1).astype(int)
                p = pred[p_key][:, h, i].reshape(-1)
                hard = (p >= 0.5).astype(int)
                rows.append({
                    "model": model,
                    "horizon_step": h + 1,
                    "lead_minutes": 15 * (h + 1),
                    "threshold": gamma,
                    "direction": direction,
                    "brier": float(brier_score_loss(y, p)),
                    "auprc": safe_score(average_precision_score, y, p),
                    "auroc": safe_score(roc_auc_score, y, p),
                    "f1": float(f1_score(y, hard, zero_division=0)),
                    "precision": float(precision_score(y, hard, zero_division=0)),
                    "recall": float(recall_score(y, hard, zero_division=0)),
                    "ece": ece_score(y, p),
                    "n": int(len(y)),
                    "events": int(y.sum()),
                })
    return rows


def trajectory_rows(model, split, pred):
    rows = []
    y = split["y_power"]
    p = pred["power_mean"]
    groups = {
        "all": np.ones(y.shape[1], dtype=bool),
        "0-1h": np.arange(y.shape[1]) < 4,
        "1-3h": (np.arange(y.shape[1]) >= 4) & (np.arange(y.shape[1]) < 12),
        "3-6h": np.arange(y.shape[1]) >= 12,
    }
    denom = max(float(np.nanmax(y) - np.nanmin(y)), 1e-6)
    for group_name, hmask in groups.items():
        err = p[:, hmask] - y[:, hmask]
        rmse = float(np.sqrt(np.mean(err ** 2)))
        rows.append({
            "model": model,
            "horizon_group": group_name,
            "mae": float(np.mean(np.abs(err))),
            "rmse": rmse,
            "nrmse": rmse / denom,
        })
    return rows


def trajectory_rows_by_horizon(model, split, pred):
    rows = []
    y = split["y_power"]
    p = pred["power_mean"]
    denom = max(float(np.nanmax(y) - np.nanmin(y)), 1e-6)
    for h in range(y.shape[1]):
        err = p[:, h] - y[:, h]
        rmse = float(np.sqrt(np.mean(err ** 2)))
        rows.append({
            "model": model,
            "horizon_step": h + 1,
            "lead_minutes": 15 * (h + 1),
            "mae": float(np.mean(np.abs(err))),
            "rmse": rmse,
            "nrmse": rmse / denom,
        })
    return rows


def add_bss(rows):
    ref = {}
    for row in rows:
        if row["model"] == "Persistence":
            key = (row["site_id"], row["horizon_group"], row["threshold"], row["direction"])
            ref[key] = row["brier"]
    for row in rows:
        key = (row["site_id"], row["horizon_group"], row["threshold"], row["direction"])
        if key in ref and ref[key] > 0:
            row["bss_vs_persistence"] = 1.0 - row["brier"] / ref[key]
    return rows


def main():
    root = Path(".")
    data_cfg, _, _ = load_configs(root)
    test = load_split(root / "outputs" / "prepared" / "multisite_splits.npz", "test")
    capacity_by_site = {site["site_id"]: float(site["capacity_kw"]) for site in data_cfg["sites"]}
    thresholds = data_cfg["ramp"]["thresholds"]
    metrics_dir = root / "outputs" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    all_event, by_horizon, traj, traj_horizon = [], [], [], []
    for model in ["Persistence"]:
        pred = baseline_predictions(test, thresholds, capacity_by_site, model)
        all_event.extend(event_rows(model, test, pred, thresholds))
        by_horizon.extend(event_rows_by_horizon(model, test, pred, thresholds))
        traj.extend(trajectory_rows(model, test, pred))
        traj_horizon.extend(trajectory_rows_by_horizon(model, test, pred))
    event_all_cols = ["model", "site_id", "horizon_group", "threshold", "direction", "brier", "auprc", "n", "events"]
    event_horizon_cols = ["model", "horizon_step", "lead_minutes", "threshold", "direction", "brier", "auprc", "n", "events"]
    pd.DataFrame(add_bss(all_event))[event_all_cols].to_csv(metrics_dir / "main_event_results_multisite.csv", index=False)
    pd.DataFrame(by_horizon)[event_horizon_cols].to_csv(metrics_dir / "main_event_results_by_horizon.csv", index=False)
    pd.DataFrame(traj).to_csv(metrics_dir / "trajectory_results_multisite.csv", index=False)
    pd.DataFrame(traj_horizon).to_csv(metrics_dir / "trajectory_results_by_horizon.csv", index=False)
    print("Wrote baseline multisite experiment metrics.")


if __name__ == "__main__":
    main()
