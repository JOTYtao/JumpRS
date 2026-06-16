import numpy as np


def persistence_predictions(split, thresholds, capacity_kw):
    last_x = split["X_hist"][:, -1, 0]
    last_power = split["p_prev"]
    power_mean = last_x * split["p_cs_next"]
    ramp = (power_mean - last_power) / float(capacity_kw)
    down = np.stack([(ramp <= -g).astype(float) for g in thresholds], axis=1)
    up = np.stack([(ramp >= g).astype(float) for g in thresholds], axis=1)
    return {"mean_x": last_x, "power_mean": power_mean, "down_prob": down, "up_prob": up}


def smart_persistence_predictions(split, thresholds, capacity_kw):
    hist = split["X_hist"]
    prev_x = hist[:, -2, 0]
    last_x = hist[:, -1, 0]
    trend_x = last_x + (last_x - prev_x)
    trend_x = np.clip(trend_x, 0.0, 1.2)
    power_mean = trend_x * split["p_cs_next"]
    ramp = (power_mean - split["p_prev"]) / float(capacity_kw)
    scale = max(min(thresholds), 1e-6)
    down = np.stack([np.clip((-ramp - g + scale) / scale, 0, 1) for g in thresholds], axis=1)
    up = np.stack([np.clip((ramp - g + scale) / scale, 0, 1) for g in thresholds], axis=1)
    return {"mean_x": trend_x, "power_mean": power_mean, "down_prob": down, "up_prob": up}
