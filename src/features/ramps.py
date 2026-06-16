import pandas as pd


def add_ramp_labels(df, capacity_kw, window_steps=1, thresholds=None):
    if thresholds is None:
        thresholds = [0.05, 0.10, 0.15, 0.20]
    out = df.copy()
    lag_power = out["power"].shift(int(window_steps))
    lag_valid = out["daylight_valid"].shift(int(window_steps)).fillna(False).infer_objects(copy=False).astype(bool)
    out["ramp_fraction"] = (out["power"] - lag_power) / float(capacity_kw)
    out["ramp_valid"] = out["daylight_valid"].astype(bool) & lag_valid
    for gamma in thresholds:
        key = str(gamma).replace(".", "p")
        out[f"down_{key}"] = ((out["ramp_fraction"] <= -float(gamma)) & out["ramp_valid"]).astype(int)
        out[f"up_{key}"] = ((out["ramp_fraction"] >= float(gamma)) & out["ramp_valid"]).astype(int)
    return out
