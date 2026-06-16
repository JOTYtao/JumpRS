from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd

from src.config import resolve_project_path


WEATHER_COLUMNS = ["air_temperature", "wind_speed"]
NSRDB_COLUMN_MAP = {
    "Temperature": "air_temperature",
    "Air Temperature": "air_temperature",
    "Wind Speed": "wind_speed",
}


def parse_nsrdb_psm3_csv(text: str, timezone: str) -> pd.DataFrame:
    """Parse an NSRDB PSM3 CSV response into timezone-aligned weather data."""
    lines = text.splitlines()
    header_idx = next(
        (i for i, line in enumerate(lines) if "Year" in line and "Month" in line and "Minute" in line),
        None,
    )
    if header_idx is None:
        preview = " ".join(lines[:3])[:300]
        raise ValueError(f"NSRDB response does not contain a PSM3 time-series header: {preview}")

    frame = pd.read_csv(StringIO("\n".join(lines[header_idx:])))
    required_time = ["Year", "Month", "Day", "Hour", "Minute"]
    missing_time = [col for col in required_time if col not in frame.columns]
    if missing_time:
        raise ValueError("NSRDB response is missing timestamp columns: " + ", ".join(missing_time))

    frame = frame.rename(columns=NSRDB_COLUMN_MAP)
    missing_weather = [col for col in WEATHER_COLUMNS if col not in frame.columns]
    if missing_weather:
        raise ValueError("NSRDB response is missing weather columns: " + ", ".join(missing_weather))

    timestamp = pd.to_datetime(
        frame[required_time].rename(
            columns={"Year": "year", "Month": "month", "Day": "day", "Hour": "hour", "Minute": "minute"}
        ),
        errors="coerce",
        utc=True,
    )
    if timestamp.isna().any():
        raise ValueError("NSRDB response contains unparseable timestamps.")

    weather = frame[WEATHER_COLUMNS].apply(pd.to_numeric, errors="coerce")
    weather.index = timestamp.dt.tz_convert(timezone)
    weather = weather.sort_index()
    weather = weather[~weather.index.duplicated(keep="first")]
    if weather.isna().any().any():
        raise ValueError("NSRDB weather contains missing or nonnumeric temperature/wind-speed values.")
    return weather


def site_nsrdb_file(data_cfg, site_cfg, year: int, root=".") -> Path:
    base = resolve_project_path(root, data_cfg["nsrdb"]["raw_dir"])
    return base / site_cfg["site_id"] / f"year={year}" / "nsrdb_weather.csv"


def load_site_nsrdb_weather(data_cfg, site_cfg, root=".") -> pd.DataFrame:
    nsrdb_cfg = data_cfg.get("nsrdb") or {}
    if not nsrdb_cfg.get("enabled", False):
        raise ValueError("NSRDB weather is required but nsrdb.enabled is false.")

    frames = []
    missing = []
    for year in nsrdb_cfg.get("support_years") or nsrdb_cfg.get("years") or data_cfg["split"]["years"]:
        path = site_nsrdb_file(data_cfg, site_cfg, int(year), root)
        if not path.exists():
            missing.append(str(path))
            continue
        frames.append(parse_nsrdb_psm3_csv(path.read_text(encoding="utf-8"), site_cfg["timezone"]))
    if missing:
        raise FileNotFoundError(
            "Missing real NSRDB weather files. Run scripts/download_nsrdb_weather.py first:\n"
            + "\n".join(missing)
        )
    weather = pd.concat(frames).sort_index()
    return weather[~weather.index.duplicated(keep="first")]


def align_nsrdb_weather(weather: pd.DataFrame, target_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Interpolate NSRDB weather from its native interval to measured-PV timestamps."""
    if target_index.tz is None:
        raise ValueError("Measured PV timestamps must be timezone-aware before NSRDB alignment.")
    if any(col not in weather for col in WEATHER_COLUMNS):
        raise ValueError("Aligned NSRDB input must contain air_temperature and wind_speed.")

    combined_index = weather.index.union(target_index.drop_duplicates()).sort_values()
    aligned = weather.reindex(combined_index).interpolate(method="time", limit_area="inside").reindex(target_index)
    return aligned[WEATHER_COLUMNS]
