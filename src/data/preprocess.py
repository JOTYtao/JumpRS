from pathlib import Path
import pandas as pd
from src.config import resolve_project_path
from src.data.real_data import load_real_power_table
from src.data.clearsky import (
    add_solar_position,
    compute_clear_sky_power_with_modelchain,
    validate_site_alignment,
    save_clearsky_metadata,
    require_site_metadata,
)
from src.data.nsrdb import align_nsrdb_weather, load_site_nsrdb_weather


def parse_power_dataframe(raw, data_cfg, site_cfg=None, columns_cfg=None):
    columns = columns_cfg or data_cfg["columns"]
    site = site_cfg or data_cfg["site"]
    require_site_metadata(site)
    if columns["timestamp_col"] not in raw.columns:
        raise ValueError(f"Missing timestamp column: {columns['timestamp_col']}")
    if columns["power_col"] not in raw.columns:
        raise ValueError(f"Missing measured PV power column: {columns['power_col']}")
    ts = pd.to_datetime(raw[columns["timestamp_col"]], errors="coerce")
    if ts.isna().any():
        raise ValueError("At least one timestamp could not be parsed.")
    df = raw.copy()
    df.index = ts
    df = df.rename(columns={columns["power_col"]: "power"}).drop(columns=[columns["timestamp_col"]])
    if df.index.tz is None:
        df.index = df.index.tz_localize(site["timezone"], nonexistent="shift_forward", ambiguous="NaT")
    else:
        df.index = df.index.tz_convert(site["timezone"])
    df = df[~df.index.isna()]
    df["power"] = pd.to_numeric(df["power"], errors="coerce")
    df["power"] = df["power"] * float(columns.get("power_scale_to_kw", 1.0))
    if df["power"].notna().sum() == 0:
        raise ValueError("Measured power contains no numeric observations.")
    df = df[~df.index.duplicated(keep="first")]
    # Real PVDAQ AC power can include small nighttime parasitic/offset values.
    # Keep the measured series, but enforce the physically valid generated-power domain.
    df["power"] = df["power"].clip(lower=0.0)
    clear_col = columns.get("clear_sky_power_col")
    if clear_col and clear_col in df.columns:
        df = df.rename(columns={clear_col: "P_cs"})
        df["P_cs"] = pd.to_numeric(df["P_cs"], errors="coerce")
    return df.sort_index()


def preprocess_power_frame(raw, data_cfg, site_cfg, columns_cfg=None, processed_dir=None, weather=None):
    df = parse_power_dataframe(raw, data_cfg, site_cfg=site_cfg, columns_cfg=columns_cfg)
    processed_dir = Path(processed_dir) if processed_dir is not None else None
    if processed_dir is not None:
        processed_dir.mkdir(parents=True, exist_ok=True)
    if "P_cs" not in df.columns:
        if weather is None:
            raise ValueError("Real NSRDB weather is required to calculate clear-sky power.")
        df = df.join(align_nsrdb_weather(weather, df.index))
        df = compute_clear_sky_power_with_modelchain(df, site_cfg)
        if processed_dir is not None:
            save_clearsky_metadata(site_cfg, processed_dir, computed=True, weather_source="NSRDB PSM3")
    else:
        df = add_solar_position(df, site_cfg)
        if processed_dir is not None:
            save_clearsky_metadata(site_cfg, processed_dir, computed=False)
    numeric = df.select_dtypes(include="number").resample(data_cfg["preprocessing"]["resample_rule"]).mean()
    if data_cfg["preprocessing"].get("missing_strategy") == "interpolate_short_gaps":
        limit = int(data_cfg["preprocessing"]["max_interpolation_gap"])
        numeric["power"] = numeric["power"].interpolate(limit=limit, limit_direction="both")
        numeric["P_cs"] = numeric["P_cs"].interpolate(limit=limit, limit_direction="both")
        if "solar_zenith" in numeric:
            numeric["solar_zenith"] = numeric["solar_zenith"].interpolate(limit=limit, limit_direction="both")
        if "solar_azimuth" in numeric:
            numeric["solar_azimuth"] = numeric["solar_azimuth"].interpolate(limit=limit, limit_direction="both")
    numeric = numeric.dropna(subset=["power", "P_cs", "solar_zenith"])
    numeric = numeric[~numeric.index.duplicated(keep="first")]
    validate_site_alignment(numeric, site_cfg)
    method = data_cfg["preprocessing"].get("daylight_method", "solar_zenith")
    if method != "solar_zenith":
        raise ValueError("Milestone 1 multisite preprocessing requires daylight_method: solar_zenith")
    zenith_max = float(data_cfg["preprocessing"].get("solar_zenith_max_degrees", 85.0))
    numeric["daylight_valid"] = numeric["solar_zenith"] < zenith_max
    numeric["x"] = (numeric["power"] / numeric["P_cs"].clip(lower=1e-6)).clip(lower=0.0, upper=float(data_cfg["preprocessing"]["x_max"]))
    numeric.loc[~numeric["daylight_valid"], "x"] = float("nan")
    numeric["site_id"] = site_cfg.get("site_id", site_cfg.get("name", "site"))
    if processed_dir is not None:
        compact = numeric.loc[numeric["daylight_valid"], ["x", "power", "P_cs"]]
        compact.to_csv(processed_dir / "processed_power.csv", index_label="timestamp")
    return numeric


def preprocess_real_data(data_cfg, root="."):
    raw = load_real_power_table(data_cfg, root)
    processed_dir = resolve_project_path(root, data_cfg["dataset"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    weather = load_site_nsrdb_weather(data_cfg, data_cfg["site"], root)
    return preprocess_power_frame(raw, data_cfg, data_cfg["site"], data_cfg.get("columns"), processed_dir, weather)
