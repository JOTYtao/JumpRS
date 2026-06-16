from pathlib import Path

from src.config import load_configs
from src.data.multisite import build_multisite_windows, event_frequency_by_site_horizon, save_npz_splits


def main():
    root = Path(".")
    data_cfg, _, _ = load_configs(root)
    windows, splits, split_summary, daylight_summary = build_multisite_windows(data_cfg, root)

    metrics_dir = root / "outputs" / "metrics"
    prepared_dir = root / "outputs" / "prepared"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    prepared_dir.mkdir(parents=True, exist_ok=True)

    split_summary.to_csv(metrics_dir / "split_summary.csv", index=False)
    daylight_summary.to_csv(metrics_dir / "daylight_summary_multisite.csv", index=False)
    event_frequency_by_site_horizon(splits, data_cfg["ramp"]["thresholds"]).to_csv(
        metrics_dir / "event_frequency_multisite.csv",
        index=False,
    )
    save_npz_splits(splits, prepared_dir / "multisite_splits.npz")

    print("Prepared multisite windows.")
    print(f"Total windows: {len(windows['y_x'])}")
    for split_name, split in splits.items():
        print(f"{split_name}: {len(split['y_x'])} windows")
    print("Wrote outputs/metrics/split_summary.csv")
    print("Wrote outputs/metrics/daylight_summary_multisite.csv")
    print("Wrote outputs/metrics/event_frequency_multisite.csv")
    print("Wrote outputs/prepared/multisite_splits.npz")


if __name__ == "__main__":
    main()
