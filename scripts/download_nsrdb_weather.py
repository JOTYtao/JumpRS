#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_configs
from src.data.nsrdb import parse_nsrdb_psm3_csv, site_nsrdb_file


def download_site_year(data_cfg, site_cfg, year: int, api_key: str, email: str, root: Path, force: bool) -> str:
    nsrdb_cfg = data_cfg["nsrdb"]
    output = site_nsrdb_file(data_cfg, site_cfg, year, root)
    if output.exists() and output.stat().st_size > 0 and not force:
        parse_nsrdb_psm3_csv(output.read_text(encoding="utf-8"), site_cfg["timezone"])
        return f"exists {output}"

    params = {
        "api_key": api_key,
        "wkt": f"POINT({site_cfg['longitude']} {site_cfg['latitude']})",
        "names": str(year),
        "attributes": ",".join(nsrdb_cfg["attributes"]),
        "interval": str(nsrdb_cfg["interval_minutes"]),
        "utc": str(bool(nsrdb_cfg.get("utc", True))).lower(),
        "leap_day": str(bool(nsrdb_cfg.get("leap_day", False))).lower(),
        "email": email,
        "full_name": "JumpRS research",
        "affiliation": "JumpRS",
        "mailing_list": "false",
        "reason": "academic research",
    }
    url = f"{nsrdb_cfg['endpoint']}?{urlencode(params)}"
    try:
        with urlopen(url, timeout=180) as response:
            text = response.read().decode("utf-8")
    except (HTTPError, URLError) as exc:
        raise RuntimeError(f"NSRDB download failed for {site_cfg['site_id']} {year}: {exc}") from exc

    parse_nsrdb_psm3_csv(text, site_cfg["timezone"])
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(".csv.tmp")
    temp.write_text(text, encoding="utf-8")
    temp.replace(output)
    return f"downloaded {output}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download NSRDB temperature and wind-speed data.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = Path(".")
    data_cfg, _, _ = load_configs(root)
    nsrdb_cfg = data_cfg.get("nsrdb") or {}
    if not nsrdb_cfg.get("enabled", False):
        raise SystemExit("nsrdb.enabled is false; no weather data were downloaded.")

    api_key = os.environ.get(nsrdb_cfg["api_key_env"], "").strip()
    email = os.environ.get(nsrdb_cfg["email_env"], "").strip()
    if not api_key or not email:
        raise SystemExit(
            f"Set {nsrdb_cfg['api_key_env']} and {nsrdb_cfg['email_env']} before downloading real NSRDB data."
        )

    for site_cfg in data_cfg["sites"]:
        for year in nsrdb_cfg.get("support_years") or nsrdb_cfg["years"]:
            print(download_site_year(data_cfg, site_cfg, int(year), api_key, email, root, args.force))


if __name__ == "__main__":
    main()
