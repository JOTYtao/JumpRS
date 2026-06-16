import numpy as np
import pandas as pd


def _empty_like_window_dict():
    return {
        "X_hist": [],
        "y_x": [],
        "y_power": [],
        "p_prev": [],
        "p_cs_prev": [],
        "p_cs_next": [],
        "y_down": [],
        "y_up": [],
        "ramp_valid": [],
        "time": [],
        "site_id": [],
    }


def _finalize_window_dict(out):
    if not out["X_hist"]:
        raise ValueError("No valid forecasting windows. Check daylight data length, missing values, and window settings.")
    return {
        "X_hist": np.asarray(out["X_hist"], dtype="float32"),
        "y_x": np.asarray(out["y_x"], dtype="float32"),
        "y_power": np.asarray(out["y_power"], dtype="float32"),
        "p_prev": np.asarray(out["p_prev"], dtype="float32"),
        "p_cs_prev": np.asarray(out["p_cs_prev"], dtype="float32"),
        "p_cs_next": np.asarray(out["p_cs_next"], dtype="float32"),
        "y_down": np.asarray(out["y_down"], dtype="float32"),
        "y_up": np.asarray(out["y_up"], dtype="float32"),
        "ramp_valid": np.asarray(out["ramp_valid"], dtype="bool"),
        "time": np.asarray(out["time"]),
        "site_id": np.asarray(out["site_id"]),
    }


def _is_contiguous(index, expected_step):
    if len(index) <= 1:
        return True
    diffs = pd.Series(index).diff().dropna()
    return bool((diffs == expected_step).all())


def make_forecasting_windows(df, data_cfg):
    history = int(data_cfg["window"]["history_steps"])
    horizon = int(data_cfg["window"]["forecast_steps"])
    stride = int(data_cfg["window"]["stride"])
    ramp_lag = int(data_cfg["ramp"].get("window_steps", 1))
    thresholds = data_cfg["ramp"]["thresholds"]
    feature_cols = ["x", "power", "P_cs"]
    rows = df.copy().sort_index()
    if not isinstance(rows.index, pd.DatetimeIndex):
        raise ValueError("Forecasting windows require a DatetimeIndex.")
    expected_step = pd.Timedelta(data_cfg["preprocessing"].get("resample_rule", "15min"))
    out = _empty_like_window_dict()
    for end in range(history, len(rows) - horizon + 1, stride):
        hist = rows.iloc[end-history:end]
        future = rows.iloc[end:end+horizon]
        block = rows.iloc[end-history:end+horizon]
        if not _is_contiguous(block.index, expected_step):
            continue
        if hist[feature_cols].isna().any().any():
            continue
        if future[feature_cols + ["ramp_valid", "daylight_valid"]].isna().any().any():
            continue
        if not bool(hist["daylight_valid"].all() and future["daylight_valid"].all() and future["ramp_valid"].all()):
            continue
        prev_power, prev_pcs = [], []
        for step in range(horizon):
            prev_idx = end + step - ramp_lag
            if prev_idx < 0:
                raise ValueError("Ramp lag reaches before the available history.")
            prev_row = rows.iloc[prev_idx]
            prev_power.append(float(prev_row["power"]))
            prev_pcs.append(float(prev_row["P_cs"]))
        yd, yu = [], []
        for _, target in future.iterrows():
            yd_step, yu_step = [], []
            for gamma in thresholds:
                key = str(gamma).replace(".", "p")
                yd_step.append(int(target[f"down_{key}"]))
                yu_step.append(int(target[f"up_{key}"]))
            yd.append(yd_step)
            yu.append(yu_step)
        site_id = str(future["site_id"].iloc[0]) if "site_id" in future else "site"
        out["X_hist"].append(hist[feature_cols].to_numpy(dtype="float32"))
        out["y_x"].append(future["x"].to_numpy(dtype="float32"))
        out["y_power"].append(future["power"].to_numpy(dtype="float32"))
        out["p_prev"].append(prev_power)
        out["p_cs_prev"].append(prev_pcs)
        out["p_cs_next"].append(future["P_cs"].to_numpy(dtype="float32"))
        out["y_down"].append(yd)
        out["y_up"].append(yu)
        out["ramp_valid"].append(future["ramp_valid"].to_numpy(dtype="bool"))
        out["time"].append(future.index.to_numpy())
        out["site_id"].append(site_id)
    finalized = _finalize_window_dict(out)
    if horizon == 1:
        for key in ["y_x", "y_power", "p_prev", "p_cs_prev", "p_cs_next", "ramp_valid"]:
            finalized[key] = finalized[key][:, 0]
        finalized["y_down"] = finalized["y_down"][:, 0, :]
        finalized["y_up"] = finalized["y_up"][:, 0, :]
        finalized["time"] = finalized["time"][:, 0]
    return finalized


def chronological_split(windows, split_cfg):
    n = len(windows["y_x"])
    n_train = int(n * float(split_cfg["train"]))
    n_val = int(n * float(split_cfg["validation"]))
    if n_train <= 0 or n_val <= 0 or n - n_train - n_val <= 0:
        raise ValueError("Not enough windows for chronological train/validation/test split.")
    idx = {"train": slice(0, n_train), "validation": slice(n_train, n_train+n_val), "test": slice(n_train+n_val, n)}
    return {name: {k: v[s] for k, v in windows.items()} for name, s in idx.items()}


def split_by_years(windows, split_cfg):
    times = np.asarray(windows["time"])
    first_times = times if times.ndim == 1 else times[:, 0]
    years = np.asarray([pd.Timestamp(t).year for t in first_times], dtype=int)
    sites = np.asarray(windows.get("site_id", np.array(["site"] * len(first_times))))
    split_indices = {"train": [], "validation": [], "test": []}
    summary_rows = []
    required_years = int(split_cfg.get("required_years", 3))
    train_years_count = int(split_cfg.get("train_years", 2))
    test_years_count = int(split_cfg.get("test_years", 1))
    val_fraction = float(split_cfg.get("validation_fraction_from_train_period", 0.15))
    for site_id in sorted(set(sites)):
        site_idx = np.where(sites == site_id)[0]
        site_years = sorted(set(int(years[i]) for i in site_idx))
        if len(site_years) < required_years:
            raise ValueError(f"Site {site_id} has {len(site_years)} window years; need at least {required_years}.")
        selected_years = site_years[: train_years_count + test_years_count]
        train_years = selected_years[:train_years_count]
        test_years = selected_years[train_years_count: train_years_count + test_years_count]
        train_period_idx = [i for i in site_idx if int(years[i]) in train_years]
        test_idx = [i for i in site_idx if int(years[i]) in test_years]
        train_period_idx = sorted(train_period_idx, key=lambda i: first_times[i])
        test_idx = sorted(test_idx, key=lambda i: first_times[i])
        if not train_period_idx or not test_idx:
            raise ValueError(f"Site {site_id} produced empty train or test windows.")
        n_val = max(1, int(len(train_period_idx) * val_fraction))
        train_idx = train_period_idx[:-n_val]
        val_idx = train_period_idx[-n_val:]
        if not train_idx:
            raise ValueError(f"Site {site_id} validation fraction leaves no training windows.")
        split_indices["train"].extend(train_idx)
        split_indices["validation"].extend(val_idx)
        split_indices["test"].extend(test_idx)
        summary_rows.extend([
            {"site_id": site_id, "split": "train", "years": ";".join(map(str, train_years)), "windows": len(train_idx)},
            {"site_id": site_id, "split": "validation", "years": ";".join(map(str, train_years)), "windows": len(val_idx)},
            {"site_id": site_id, "split": "test", "years": ";".join(map(str, test_years)), "windows": len(test_idx)},
        ])
    out = {}
    for split_name, idxs in split_indices.items():
        idxs = np.asarray(sorted(idxs, key=lambda i: (str(sites[i]), first_times[i])), dtype=int)
        out[split_name] = {k: v[idxs] for k, v in windows.items()}
    return out, pd.DataFrame(summary_rows)
