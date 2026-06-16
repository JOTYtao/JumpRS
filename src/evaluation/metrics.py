from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, mean_absolute_error, mean_squared_error


def safe_auprc(y, p):
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, p))


def event_metrics(y_down, y_up, pred, thresholds):
    rows = []
    for i, gamma in enumerate(thresholds):
        rows.append({
            "threshold": gamma,
            "down_brier": brier_score_loss(y_down[:, i], pred["down_prob"][:, i]),
            "down_auprc": safe_auprc(y_down[:, i], pred["down_prob"][:, i]),
            "up_brier": brier_score_loss(y_up[:, i], pred["up_prob"][:, i]),
            "up_auprc": safe_auprc(y_up[:, i], pred["up_prob"][:, i]),
        })
    return rows


def trajectory_metrics(y_power, pred_power):
    mse = mean_squared_error(y_power, pred_power)
    return {
        "mae": mean_absolute_error(y_power, pred_power),
        "rmse": float(np.sqrt(mse)),
    }


def event_frequency(split, thresholds):
    rows = []
    for name, data in split.items():
        for i, gamma in enumerate(thresholds):
            rows.append({"split": name, "direction": "down", "threshold": gamma, "n": len(data["y_x"]), "events": int(data["y_down"][:, i].sum()), "event_rate": float(data["y_down"][:, i].mean())})
            rows.append({"split": name, "direction": "up", "threshold": gamma, "n": len(data["y_x"]), "events": int(data["y_up"][:, i].sum()), "event_rate": float(data["y_up"][:, i].mean())})
    return pd.DataFrame(rows)
