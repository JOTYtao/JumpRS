from pathlib import Path

import numpy as np
import pandas as pd

from src.config import resolve_project_path
from src.data.clearsky import require_site_metadata
from src.data.preprocess import preprocess_power_frame
from src.data.nsrdb import load_site_nsrdb_weather
from src.data.real_data import RealDataRequiredError, read_table
from src.features.ramps import add_ramp_labels
from src.features.windows import make_forecasting_windows, split_by_years


def _site_columns(data_cfg, site_cfg):
    columns = dict(data_cfg.get("columns", {}))
    columns.update(site_cfg.get("columns", {}) or {})
    return columns


def _site_key(site_cfg):
    return site_cfg.get("site_id") or site_cfg.get("name") or "site"


def validate_multisite_config(data_cfg):
    if data_cfg["dataset"].get("use_synthetic_data") is not False:
        raise RealDataRequiredError("Synthetic PV power is prohibited for multisite experiments.")
    sites = data_cfg.get("sites") or []
    if len(sites) != 3:
        raise ValueError(f"Multisite experiment requires exactly 3 sites; found {len(sites)}.")
    for site in sites:
        try:
            require_site_metadata(site)
        except ValueError as exc:
            raise ValueError(f"Site {_site_key(site)} is missing critical metadata: {exc}") from exc
        if site.get("allow_capacity_inference"):
            raise ValueError(f"Site {_site_key(site)} has allow_capacity_inference=true; capacity must be explicit.")
    return sites


def configured_site_files(site_cfg, root="."):
    files = []
    for item in site_cfg.get("local_files", []) or []:
        p = resolve_project_path(root, item)
        if not p.exists():
            raise FileNotFoundError(f"Configured real-data file does not exist for site {_site_key(site_cfg)}: {p}")
        files.append(p)
    raw_dir = site_cfg.get("raw_dir")
    if raw_dir:
        p = resolve_project_path(root, raw_dir)
        if p.exists():
            files.extend(sorted(x for x in p.rglob("*") if x.is_file() and x.suffix.lower() in {".csv", ".parquet"}))
    seen = []
    for p in files:
        if p not in seen:
            seen.append(p)
    if not seen:
        raise RealDataRequiredError(
            f"No real measured PV files found for site {_site_key(site_cfg)}. "
            "Populate site.local_files or site.raw_dir with CSV/parquet files."
        )
    return seen


def load_site_power_table(site_cfg, root="."):
    frames = [read_table(path) for path in configured_site_files(site_cfg, root)]
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]


def preprocess_multisite_data(data_cfg, root="."):
    sites = validate_multisite_config(data_cfg)
    root = Path(root)
    processed_base = resolve_project_path(root, data_cfg["dataset"]["processed_dir"])
    processed = []
    daylight_rows = []
    for site_cfg in sites:
        site_id = _site_key(site_cfg)
        raw = load_site_power_table(site_cfg, root)
        weather = load_site_nsrdb_weather(data_cfg, site_cfg, root)
        site_processed_dir = processed_base / site_id
        df = preprocess_power_frame(
            raw,
            data_cfg,
            site_cfg,
            columns_cfg=_site_columns(data_cfg, site_cfg),
            processed_dir=site_processed_dir,
            weather=weather,
        )
        df["site_id"] = site_id
        processed.append(df)
        daylight_rows.append({
            "site_id": site_id,
            "samples_total": int(len(df)),
            "daylight_samples": int(df["daylight_valid"].sum()),
            "daylight_fraction": float(df["daylight_valid"].mean()) if len(df) else 0.0,
            "first_timestamp": df.index.min().isoformat() if len(df) else "",
            "last_timestamp": df.index.max().isoformat() if len(df) else "",
        })
    return processed, pd.DataFrame(daylight_rows)


def build_multisite_windows(data_cfg, root="."):
    processed_frames, daylight_summary = preprocess_multisite_data(data_cfg, root)
    site_windows = []
    thresholds = data_cfg["ramp"]["thresholds"]
    for df in processed_frames:
        site_id = str(df["site_id"].iloc[0])
        site_cfg = next(s for s in data_cfg["sites"] if _site_key(s) == site_id)
        labeled = add_ramp_labels(
            df,
            float(site_cfg["capacity_kw"]),
            data_cfg["ramp"]["window_steps"],
            thresholds,
        )
        windows = make_forecasting_windows(labeled, data_cfg)
        windows["capacity_kw"] = np.full_like(windows["p_cs_next"], float(site_cfg["capacity_kw"]), dtype="float32")
        site_windows.append(windows)
    windows = concatenate_window_sets(site_windows)
    splits, split_summary = split_by_years(windows, data_cfg["split"])
    return windows, splits, split_summary, daylight_summary


def concatenate_window_sets(window_sets):
    if not window_sets:
        raise ValueError("No site window sets were provided.")
    keys = window_sets[0].keys()
    return {key: np.concatenate([w[key] for w in window_sets], axis=0) for key in keys}


def save_npz_splits(splits, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    for split_name, split in splits.items():
        for key, value in split.items():
            payload[f"{split_name}__{key}"] = value
    np.savez_compressed(output_path, **payload)


def event_frequency_by_site_horizon(splits, thresholds):
    rows = []
    for split_name, split in splits.items():
        y_down = split["y_down"]
        y_up = split["y_up"]
        sites = split["site_id"]
        if y_down.ndim == 2:
            y_down = y_down[:, None, :]
            y_up = y_up[:, None, :]
        for site_id in sorted(set(sites)):
            mask = sites == site_id
            for h in range(y_down.shape[1]):
                for i, gamma in enumerate(thresholds):
                    down = y_down[mask, h, i]
                    up = y_up[mask, h, i]
                    rows.append({
                        "split": split_name,
                        "site_id": site_id,
                        "horizon_step": h + 1,
                        "lead_minutes": 15 * (h + 1),
                        "direction": "down",
                        "threshold": gamma,
                        "n": int(len(down)),
                        "events": int(down.sum()),
                        "event_rate": float(down.mean()) if len(down) else 0.0,
                    })
                    rows.append({
                        "split": split_name,
                        "site_id": site_id,
                        "horizon_step": h + 1,
                        "lead_minutes": 15 * (h + 1),
                        "direction": "up",
                        "threshold": gamma,
                        "n": int(len(up)),
                        "events": int(up.sum()),
                        "event_rate": float(up.mean()) if len(up) else 0.0,
                    })
    return pd.DataFrame(rows)
