from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(".")
PREDICTIONS = ROOT / "outputs" / "predictions" / "raw_test_sample_predictions_multisite.csv"
OUT_DIR = ROOT / "outputs" / "optimization"

RETAINED_MODELS = [
    "JumpRS",
    "QuantileGRU",
    "MC Dropout",
    "PatchTST",
    "iTransformer",
    "TimesNet",
    "Persistence",
    "TimeDiff-style",
    "NsDiff-style",
]


def _aggregate_prediction_metrics() -> pd.DataFrame:
    usecols = [
        "model",
        "site_id",
        "horizon_step",
        "lead_minutes",
        "actual_power_kw",
        "predicted_power_kw",
        "crps_kw",
    ]
    chunks = []
    for chunk in pd.read_csv(PREDICTIONS, usecols=usecols, chunksize=250_000):
        chunk = chunk[chunk["model"].isin(RETAINED_MODELS)].copy()
        err = chunk["predicted_power_kw"] - chunk["actual_power_kw"]
        chunk["abs_err"] = err.abs()
        chunk["sq_err"] = err * err
        chunk["crps_eval"] = chunk["crps_kw"].where(chunk["crps_kw"].notna(), chunk["abs_err"])
        grouped = (
            chunk.groupby(["model", "site_id", "horizon_step", "lead_minutes"], observed=True)
            .agg(
                n=("abs_err", "size"),
                mae_sum=("abs_err", "sum"),
                sq_err_sum=("sq_err", "sum"),
                crps_sum=("crps_eval", "sum"),
            )
            .reset_index()
        )
        chunks.append(grouped)
    totals = pd.concat(chunks, ignore_index=True)
    totals = (
        totals.groupby(["model", "site_id", "horizon_step", "lead_minutes"], observed=True)
        .sum(numeric_only=True)
        .reset_index()
    )
    totals["mae"] = totals["mae_sum"] / totals["n"]
    totals["rmse"] = np.sqrt(totals["sq_err_sum"] / totals["n"])
    totals["crps"] = totals["crps_sum"] / totals["n"]
    return totals[["model", "site_id", "horizon_step", "lead_minutes", "n", "mae", "rmse", "crps"]]


def _gap_table(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (site_id, horizon_step), group in metrics.groupby(["site_id", "horizon_step"], observed=True):
        jump = group[group["model"] == "JumpRS"].iloc[0]
        row = {
            "site_id": site_id,
            "horizon_step": int(horizon_step),
            "lead_minutes": int(jump["lead_minutes"]),
            "n": int(jump["n"]),
        }
        for metric in ["mae", "rmse", "crps"]:
            best = group.loc[group[metric].idxmin()]
            jump_value = float(jump[metric])
            best_value = float(best[metric])
            row[f"jumprs_{metric}"] = jump_value
            row[f"best_{metric}"] = best_value
            row[f"best_{metric}_model"] = best["model"]
            row[f"{metric}_gap"] = jump_value - best_value
            row[f"{metric}_win"] = bool(best["model"] == "JumpRS" or np.isclose(jump_value, best_value, atol=1e-8))
        row["all_three_win"] = bool(row["mae_win"] and row["rmse_win"] and row["crps_win"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["site_id", "horizon_step"]).reset_index(drop=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = _aggregate_prediction_metrics()
    metrics.to_csv(OUT_DIR / "site_lead_model_metrics.csv", index=False)
    gaps = _gap_table(metrics)
    gaps.to_csv(OUT_DIR / "jumprs_site_lead_gap_audit.csv", index=False)

    summary_rows = []
    for metric in ["mae", "rmse", "crps"]:
        wins = int(gaps[f"{metric}_win"].sum())
        total = int(len(gaps))
        worst = gaps.loc[gaps[f"{metric}_gap"].idxmax()]
        summary_rows.append(
            {
                "metric": metric,
                "jump_rs_wins": wins,
                "total_site_leads": total,
                "win_rate": wins / total,
                "worst_gap": float(worst[f"{metric}_gap"]),
                "worst_gap_site_id": worst["site_id"],
                "worst_gap_horizon_step": int(worst["horizon_step"]),
                "worst_gap_lead_minutes": int(worst["lead_minutes"]),
                "worst_gap_best_model": worst[f"best_{metric}_model"],
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_DIR / "jumprs_site_lead_gap_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"All-metric site/lead wins: {int(gaps['all_three_win'].sum())}/{len(gaps)}")
    print(f"Wrote {OUT_DIR / 'jumprs_site_lead_gap_audit.csv'}")


if __name__ == "__main__":
    main()
