#!/usr/bin/env python3
from __future__ import annotations

import argparse
import calendar
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen
import xml.etree.ElementTree as ET


BUCKET_URL = "https://oedi-data-lake.s3.amazonaws.com"
S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


def list_keys(prefix: str) -> list[str]:
    keys: list[str] = []
    token = None
    while True:
        url = f"{BUCKET_URL}/?list-type=2&prefix={prefix}&max-keys=1000"
        if token:
            url += f"&continuation-token={token}"
        with urlopen(url, timeout=60) as resp:
            root = ET.fromstring(resp.read())
        keys.extend(node.text for node in root.findall(".//s3:Key", S3_NS) if node.text)
        token_node = root.find(".//s3:NextContinuationToken", S3_NS)
        if token_node is None or not token_node.text:
            break
        token = token_node.text
    return keys


def download_key(key: str, output_root: Path, force: bool = False) -> tuple[str, str]:
    parts = key.split("/")
    system = parts[3].split("=")[1]
    year = parts[4].split("=")[1]
    month = parts[5].split("=")[1]
    day = parts[6].split("=")[1]
    out_dir = output_root / f"pvdaq_system_{system}" / f"year={year}" / f"month={month}" / f"day={day}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / parts[-1]
    if out_path.exists() and out_path.stat().st_size > 0 and not force:
        return key, "exists"
    try:
        with urlopen(f"{BUCKET_URL}/{key}", timeout=120) as resp:
            data = resp.read()
    except (HTTPError, URLError) as exc:
        return key, f"error:{exc}"
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(out_path)
    return key, "downloaded"


def collect_daily_keys(system_ids: list[int], years: list[int]) -> list[str]:
    keys: list[str] = []
    for system_id in system_ids:
        for year in years:
            prefix = f"pvdaq/csv/pvdata/system_id={system_id}/year={year}/"
            year_keys = [k for k in list_keys(prefix) if k.endswith(".csv")]
            if not year_keys:
                print(f"Warning: no CSV objects found for system={system_id}, year={year}")
                continue
            print(f"Found {len(year_keys)} files for system={system_id}, year={year}")
            keys.extend(year_keys)
    return sorted(set(keys))


def main() -> None:
    parser = argparse.ArgumentParser(description="Download real PVDAQ daily CSV files from public OEDI S3.")
    parser.add_argument("--systems", nargs="+", type=int, default=[4, 10, 34])
    parser.add_argument("--years", nargs="+", type=int, default=[2011, 2012, 2013])
    parser.add_argument("--output-root", default="data/raw/multisite")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    keys = collect_daily_keys(args.systems, args.years)
    if not keys:
        raise SystemExit("No PVDAQ CSV keys found. Stopping without creating fake data.")
    print(f"Found {len(keys)} CSV objects for systems={args.systems}, years={args.years}")
    counts = {"exists": 0, "downloaded": 0, "error": 0}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download_key, key, output_root, args.force) for key in keys]
        for i, fut in enumerate(as_completed(futures), 1):
            key, status = fut.result()
            bucket = status if status in counts else "error"
            counts[bucket] += 1
            if i % 100 == 0 or bucket == "error":
                print(f"{i}/{len(keys)} {status} {key}")
    print(f"Download summary: {counts}")
    if counts["error"]:
        raise SystemExit("Some downloads failed; inspect the log above before running experiments.")


if __name__ == "__main__":
    main()
