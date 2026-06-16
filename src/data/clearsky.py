from pathlib import Path
import json
import pandas as pd


REQUIRED_SITE_FIELDS = ["latitude", "longitude", "timezone", "capacity_kw"]


def require_site_metadata(site_cfg):
    missing = [k for k in REQUIRED_SITE_FIELDS if site_cfg.get(k) in (None, "")]
    if missing:
        raise ValueError("Missing required site metadata: " + ", ".join(missing))
    if site_cfg.get("allow_capacity_inference"):
        raise ValueError("site.allow_capacity_inference must remain false for real-data validation.")


def ensure_timezone_index(df, timezone):
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Data must be indexed by timestamps before clear-sky construction.")
    out = df.copy().sort_index()
    if out.index.tz is None:
        out.index = out.index.tz_localize(timezone, nonexistent="shift_forward", ambiguous="NaT")
    else:
        out.index = out.index.tz_convert(timezone)
    if out.index.isna().any():
        out = out[~out.index.isna()]
    if out.index.has_duplicates:
        out = out[~out.index.duplicated(keep="first")]
    return out


def compute_clear_sky_irradiance_with_pvlib(df, site_config):
    """Use pvlib.location.Location and exact site metadata to compute clear-sky GHI/DNI/DHI."""
    require_site_metadata(site_config)
    import pvlib
    out = ensure_timezone_index(df, site_config["timezone"])
    loc = pvlib.location.Location(
        latitude=float(site_config["latitude"]),
        longitude=float(site_config["longitude"]),
        tz=site_config["timezone"],
        altitude=float(site_config.get("altitude") or 0.0),
    )
    cs = loc.get_clearsky(out.index, model="ineichen")
    out["clear_sky_ghi"] = cs["ghi"].clip(lower=0.0)
    out["clear_sky_dni"] = cs["dni"].clip(lower=0.0)
    out["clear_sky_dhi"] = cs["dhi"].clip(lower=0.0)
    return out


def add_solar_position(df, site_config):
    """Add site-aligned solar position fields used for daylight filtering."""
    require_site_metadata(site_config)
    import pvlib
    out = ensure_timezone_index(df, site_config["timezone"])
    loc = pvlib.location.Location(
        latitude=float(site_config["latitude"]),
        longitude=float(site_config["longitude"]),
        tz=site_config["timezone"],
        altitude=float(site_config.get("altitude") or 0.0),
    )
    solpos = loc.get_solarposition(out.index)
    out["solar_zenith"] = solpos["apparent_zenith"]
    out["solar_azimuth"] = solpos["azimuth"]
    return out


def compute_clear_sky_power_with_modelchain(df, site_config):
    """Convert clear-sky irradiance into clear-sky AC power with a PVWatts-style pvlib workflow."""
    require_site_metadata(site_config)
    import pvlib
    out = compute_clear_sky_irradiance_with_pvlib(df, site_config)
    loc = pvlib.location.Location(
        latitude=float(site_config["latitude"]),
        longitude=float(site_config["longitude"]),
        tz=site_config["timezone"],
        altitude=float(site_config.get("altitude") or 0.0),
    )
    tilt = float(site_config["tilt"] if site_config.get("tilt") is not None else abs(float(site_config["latitude"])))
    azimuth = float(site_config["azimuth"] if site_config.get("azimuth") is not None else 180.0)
    solpos = loc.get_solarposition(out.index)
    out["solar_zenith"] = solpos["apparent_zenith"]
    out["solar_azimuth"] = solpos["azimuth"]
    dni_extra = pvlib.irradiance.get_extra_radiation(out.index)
    airmass = loc.get_airmass(out.index)["airmass_relative"]
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt,
        surface_azimuth=azimuth,
        dni=out["clear_sky_dni"],
        ghi=out["clear_sky_ghi"],
        dhi=out["clear_sky_dhi"],
        solar_zenith=solpos["apparent_zenith"],
        solar_azimuth=solpos["azimuth"],
        dni_extra=dni_extra,
        airmass=airmass,
        model="haydavies",
    )
    missing_weather = [col for col in ("air_temperature", "wind_speed") if col not in out]
    if missing_weather:
        raise ValueError(
            "Clear-sky power calculation requires aligned real NSRDB weather columns: "
            + ", ".join(missing_weather)
        )
    daylight = out["solar_zenith"] < 85.0
    missing_daylight = out.loc[daylight, ["air_temperature", "wind_speed"]].isna().sum()
    if missing_daylight.any():
        raise ValueError(
            "NSRDB weather does not cover all daylight PV timestamps: "
            + ", ".join(f"{key}={value}" for key, value in missing_daylight.items() if value)
        )
    # Boundary gaps can occur only at night after UTC-to-local conversion. These
    # values do not affect zero-irradiance P_cs, but pvlib still requires finite inputs.
    weather = out[["air_temperature", "wind_speed"]].ffill().bfill()
    temp_cell = pvlib.temperature.faiman(
        poa["poa_global"].clip(lower=0.0),
        weather["air_temperature"],
        wind_speed=weather["wind_speed"].clip(lower=0.0),
    )
    capacity_kw = float(site_config["capacity_kw"])
    pdc0 = capacity_kw * 1000.0
    pdc = pvlib.pvsystem.pvwatts_dc(poa["poa_global"].clip(lower=0.0), temp_cell, pdc0=pdc0, gamma_pdc=-0.003)
    pac_w = pvlib.inverter.pvwatts(pdc, pdc0, eta_inv_nom=0.96).clip(lower=0.0)
    out["P_cs"] = (pac_w / 1000.0).clip(lower=0.0)
    return out


def align_power_and_clear_sky(df):
    """Ensure measured power and clear-sky power share timestamp index, timezone, sampling interval, and no shift."""
    if "power" not in df or "P_cs" not in df:
        raise ValueError("Both measured power and P_cs are required.")
    if df.index.tz is None:
        raise ValueError("Timestamp index must be timezone-aware.")
    out = df.sort_index()
    if out.index.has_duplicates:
        raise ValueError("Duplicate timestamps detected.")
    diffs = out.index.to_series().diff().dropna()
    if diffs.empty:
        raise ValueError("At least two timestamps are required for alignment checks.")
    irregular = (diffs != diffs.mode().iloc[0]).mean()
    if irregular > 0.05:
        raise ValueError(f"Sampling interval too irregular for alignment: {irregular:.1%} irregular.")
    return out


def validate_site_alignment(df, site_config):
    """Validate physical consistency of P_cs with the configured site."""
    require_site_metadata(site_config)
    out = align_power_and_clear_sky(df)
    capacity_kw = float(site_config["capacity_kw"])
    if out["P_cs"].max() > 1.25 * capacity_kw:
        raise ValueError("P_cs exceeds configured capacity by more than 25%.")
    night_fraction = (out["P_cs"] <= max(out["P_cs"].max() * 1e-3, 1e-6)).mean()
    if night_fraction < 0.02:
        raise ValueError("P_cs has too few near-zero nighttime samples; check timezone/location.")
    peak = out[out["P_cs"] >= out["P_cs"].quantile(0.99)]
    if not peak.empty:
        peak_hour = peak.index.hour + peak.index.minute / 60.0
        if ((peak_hour < 8.0) | (peak_hour > 16.5)).mean() > 0.5:
            raise ValueError("P_cs peak timing is implausible; check timezone/site alignment.")
    if out["power"].max() > 1.5 * capacity_kw:
        raise ValueError("Measured power exceeds configured capacity by more than 50%; check units or capacity_kw.")
    daylight_peak = out.loc[out["P_cs"] > 0.2 * capacity_kw, "power"].max()
    if pd.notna(daylight_peak) and daylight_peak < 0.05 * capacity_kw:
        raise ValueError("Measured daylight peak is below 5% of configured capacity; check power units, column, or capacity_kw.")
    return True


def save_clearsky_metadata(site_config, processed_dir, computed=True, weather_source=None):
    processed = Path(processed_dir)
    processed.mkdir(parents=True, exist_ok=True)
    meta = {
        "latitude": site_config.get("latitude"),
        "longitude": site_config.get("longitude"),
        "altitude": site_config.get("altitude"),
        "timezone": site_config.get("timezone"),
        "clear_sky_model": "ineichen",
        "pv_model": "pvlib_pvwatts" if computed else "provider_supplied",
        "capacity_kw": site_config.get("capacity_kw"),
        "tilt": site_config.get("tilt"),
        "azimuth": site_config.get("azimuth"),
        "module_parameters_source": "PVWatts approximation" if computed else "dataset/provider",
        "inverter_parameters_source": "PVWatts approximation" if computed else "dataset/provider",
        "timestamp_alignment_checked": True,
        "daylight_filter": "solar_zenith < 85 degrees",
        "temperature_source": weather_source if computed else None,
        "wind_speed_source": weather_source if computed else None,
    }
    (processed / "clearsky_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
