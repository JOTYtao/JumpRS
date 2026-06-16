from pathlib import Path
import pandas as pd
from src.config import resolve_project_path


class RealDataRequiredError(RuntimeError):
    pass


def configured_real_files(data_cfg, root="."):
    if data_cfg["dataset"].get("use_synthetic_data") is not False:
        raise RealDataRequiredError("Synthetic data is prohibited for training/validation/testing.")
    files = []
    for item in data_cfg.get("download", {}).get("local_files", []) or []:
        p = resolve_project_path(root, item)
        if not p.exists():
            raise FileNotFoundError(f"Configured real-data file does not exist: {p}")
        files.append(p)
    raw_dir = resolve_project_path(root, data_cfg["dataset"]["raw_dir"])
    if raw_dir.exists():
        files.extend(sorted(p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".csv", ".parquet"}))
    seen = []
    for p in files:
        if p not in seen:
            seen.append(p)
    if not seen:
        raise RealDataRequiredError(
            "No real measured PV power dataset configured. Put CSV/parquet files in data/raw/ or set download.local_files in config/data.yaml."
        )
    return seen


def read_table(path):
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    raise RealDataRequiredError(f"Unsupported file format for real data: {path}")


def load_real_power_table(data_cfg, root="."):
    frames = [read_table(p) for p in configured_real_files(data_cfg, root)]
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
