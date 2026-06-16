from pathlib import Path

import pandas as pd

from src.config import load_configs
from src.models.multistep_baselines import predict_diffusion, train_diffusion
from src.run_multisite_baselines import event_rows_by_horizon, load_split, trajectory_rows_by_horizon
from src.run_multisite_experiments import _sample_pred_from_raw, crps_rows_by_horizon


def load_splits(root):
    path = root / "outputs" / "prepared" / "multisite_splits.npz"
    return {name: load_split(path, name) for name in ["train", "validation", "test"]}


def main():
    root = Path(".")
    data_cfg, _, train_cfg = load_configs(root)
    splits = load_splits(root)
    output = root / "outputs" / "tuning" / "diffusion"
    output.mkdir(parents=True, exist_ok=True)
    summary = []
    for name in ["TimeDiff-style", "NsDiff-style"]:
        model, local, history, device = train_diffusion(name, splits, train_cfg)
        raw = predict_diffusion(model, local["validation"], device, n_samples=50)
        pred = _sample_pred_from_raw(raw, local["validation"], data_cfg["ramp"]["thresholds"])
        trajectory = pd.DataFrame(trajectory_rows_by_horizon(name, local["validation"], pred))
        crps = pd.DataFrame(crps_rows_by_horizon(name, local["validation"], pred))
        events = pd.DataFrame(event_rows_by_horizon(name, local["validation"], pred, data_cfg["ramp"]["thresholds"]))
        pd.DataFrame(history).to_csv(output / f"{name}_history.csv", index=False)
        trajectory.to_csv(output / f"{name}_trajectory_by_horizon.csv", index=False)
        crps.to_csv(output / f"{name}_crps_by_horizon.csv", index=False)
        events.to_csv(output / f"{name}_event_by_horizon.csv", index=False)
        summary.append({
            "model": name,
            "validation_rmse": trajectory["rmse"].mean(),
            "validation_mae": trajectory["mae"].mean(),
            "validation_crps": crps["crps"].mean(),
            "validation_brier": events["brier"].mean(),
            "validation_auprc": events["auprc"].mean(),
            "validation_auroc": events["auroc"].mean(),
            "validation_f1": events["f1"].mean(),
            "validation_ece": events["ece"].mean(),
        })
    pd.DataFrame(summary).to_csv(output / "diffusion_validation_summary.csv", index=False)
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
