from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(".")
METRICS = ROOT / "outputs" / "metrics"
PRED = ROOT / "outputs" / "predictions" / "raw_test_sample_predictions_multisite.csv"
FIG = ROOT / "outputs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

MODEL_ORDER = [
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
KEY_MODELS = [
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
COLORS = {
    "JumpRS": "#D55E00",
    "QuantileGRU": "#009E73",
    "MC Dropout": "#8C564B",
    "PatchTST": "#CC79A7",
    "iTransformer": "#F0E442",
    "TimesNet": "#56B4E9",
    "Persistence": "#000000",
    "TimeDiff-style": "#0072B2",
    "NsDiff-style": "#999999",
}


def _setup():
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "font.family": "DejaVu Serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _ordered(df):
    return df.assign(model=pd.Categorical(df["model"], MODEL_ORDER, ordered=True)).sort_values("model")


def _metric_path(name):
    return METRICS / name


def _prediction_frames(usecols):
    base = pd.read_csv(PRED, usecols=usecols)
    base = base[base["model"].isin(KEY_MODELS)]
    return base


def plot_trajectory_leaderboard():
    df = pd.read_csv(_metric_path("trajectory_results_multisite.csv"))
    df = df[df["horizon_group"] == "all"].sort_values("rmse", ascending=True)
    fig, ax = plt.subplots(figsize=(3.45, 2.55))
    ax.barh(df["model"], df["rmse"], color=[COLORS.get(m, "#999999") for m in df["model"]])
    ax.invert_yaxis()
    ax.set_xlabel("RMSE (kW)")
    ax.set_ylabel("")
    ax.set_title("Trajectory Forecasting Accuracy")
    fig.tight_layout()
    fig.savefig(FIG / "trajectory_rmse_leaderboard_multisite.pdf")
    plt.close(fig)

    df = df.sort_values("nrmse", ascending=True)
    fig, ax = plt.subplots(figsize=(3.45, 2.55))
    ax.barh(df["model"], df["nrmse"], color=[COLORS.get(m, "#999999") for m in df["model"]])
    ax.invert_yaxis()
    ax.set_xlabel("nRMSE")
    ax.set_ylabel("")
    ax.set_title("Normalized Trajectory Forecasting Accuracy")
    fig.tight_layout()
    fig.savefig(FIG / "trajectory_nrmse_leaderboard_multisite.pdf")
    plt.close(fig)


def plot_event_bar_gamma005():
    df = pd.read_csv(_metric_path("main_event_results_multisite.csv"))
    df = df[(df["site_id"] == "ALL") & (df["horizon_group"] == "all") & (np.isclose(df["threshold"], 0.05))]
    df = df[df["model"].isin(KEY_MODELS)]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.65), sharey=False)
    for ax, metric, title in zip(axes, ["brier", "auprc"], ["Brier Score", "AUPRC"]):
        pivot = df.pivot_table(index="model", columns="direction", values=metric, aggfunc="mean").reindex(KEY_MODELS)
        x = np.arange(len(pivot))
        width = 0.38
        ax.bar(x - width / 2, pivot["down"], width, label="Down-ramp", color="#0072B2")
        ax.bar(x + width / 2, pivot["up"], width, label="Up-ramp", color="#E69F00")
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index, rotation=35, ha="right")
        ax.set_title(title + " at $\\gamma=0.05$")
        ax.set_ylabel(metric.upper() if metric != "auprc" else "AUPRC")
    axes[0].legend(frameon=False, ncol=1)
    fig.tight_layout()
    fig.savefig(FIG / "event_brier_auprc_gamma005_multisite.pdf")
    plt.close(fig)


def plot_event_horizon():
    df = pd.read_csv(_metric_path("main_event_results_by_horizon.csv"))
    df = df[(np.isclose(df["threshold"], 0.05)) & (df["model"].isin(KEY_MODELS))]
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.25), sharex=True)
    for row, metric in enumerate(["brier", "auprc"]):
        for col, direction in enumerate(["down", "up"]):
            ax = axes[row, col]
            sub = df[df["direction"] == direction]
            for model in KEY_MODELS:
                s = sub[sub["model"] == model].sort_values("lead_minutes")
                ax.plot(s["lead_minutes"] / 60.0, s[metric], label=model, lw=1.25, color=COLORS.get(model))
            ax.set_title(f"{direction.capitalize()}-ramp {metric.upper() if metric != 'auprc' else 'AUPRC'}")
            ax.set_ylabel(metric.upper() if metric != "auprc" else "AUPRC")
            ax.set_xlabel("Lead time (h)")
    axes[0, 0].legend(frameon=False, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(FIG / "event_metrics_by_horizon.pdf")
    plt.close(fig)


def plot_trajectory_horizon():
    df = pd.read_csv(_metric_path("trajectory_results_by_horizon.csv"))
    df = df[df["model"].isin(KEY_MODELS)]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.65), sharex=True)
    for ax, metric in zip(axes, ["rmse", "mae"]):
        for model in KEY_MODELS:
            s = df[df["model"] == model].sort_values("lead_minutes")
            ax.plot(s["lead_minutes"] / 60.0, s[metric], label=model, lw=1.25, color=COLORS.get(model))
        ax.set_title(metric.upper() + " by Forecast Lead")
        ax.set_xlabel("Lead time (h)")
        ax.set_ylabel(metric.upper() + " (kW)")
    axes[0].legend(frameon=False, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(FIG / "trajectory_metrics_by_horizon.pdf")
    plt.close(fig)

    for metric, ylabel, fname in [
        ("rmse", "RMSE (kW)", "rmse_by_lead_time_multisite.pdf"),
        ("mae", "MAE (kW)", "mae_by_lead_time_multisite.pdf"),
    ]:
        fig, ax = plt.subplots(figsize=(3.45, 2.65))
        for model in KEY_MODELS:
            s = df[df["model"] == model].sort_values("lead_minutes")
            ax.plot(s["lead_minutes"] / 60.0, s[metric], label=model, lw=1.35, color=COLORS.get(model))
        ax.set_xlabel("Lead time (h)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{metric.upper()} versus Forecast Lead Time")
        ax.legend(frameon=False, fontsize=6, ncol=2)
        fig.tight_layout()
        fig.savefig(FIG / fname)
        plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(3.45, 4.5), sharex=True)
    for ax, metric, ylabel in zip(axes, ["rmse", "mae"], ["RMSE (kW)", "MAE (kW)"]):
        for model in KEY_MODELS:
            s = df[df["model"] == model].sort_values("lead_minutes")
            ax.plot(s["lead_minutes"] / 60.0, s[metric], label=model, lw=1.25, color=COLORS.get(model))
        ax.set_ylabel(ylabel)
        ax.set_title(f"{metric.upper()} by Lead Time")
    axes[-1].set_xlabel("Lead time (h)")
    axes[0].legend(frameon=False, fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "rmse_mae_by_lead_time_multisite.pdf")
    plt.close(fig)


def plot_crps_selected_horizons():
    df = pd.read_csv(_metric_path("crps_results_by_horizon.csv"))
    df = df[
        df["model"].isin(KEY_MODELS)
        & df["horizon_step"].isin([1, 4, 8, 16])
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.85))
    for ax, step in zip(axes.flat, [1, 4, 8, 16]):
        sub = df[df["horizon_step"] == step].sort_values("crps", ascending=False)
        ax.barh(
            sub["model"],
            sub["crps"],
            color=[COLORS.get(model, "#999999") for model in sub["model"]],
        )
        ax.set_title(f"Step {step} ({int(sub['lead_minutes'].iloc[0])} min)")
        ax.set_xlabel("CRPS (kW)")
        ax.set_ylabel("")
        ax.grid(axis="x", alpha=0.25)
        ax.grid(axis="y", visible=False)
        for y, value in enumerate(sub["crps"]):
            ax.text(value, y, f" {value:.3f}", va="center", fontsize=6)
        ax.set_xlim(0, sub["crps"].max() * 1.14)
    fig.suptitle("Probabilistic forecast CRPS at selected lead times", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(FIG / "crps_bar_selected_horizons.pdf")
    plt.close(fig)


def plot_case_study():
    usecols = [
        "model",
        "sample_id",
        "site_id",
        "target_time",
        "horizon_step",
        "actual_power_kw",
        "p_cs_next_kw",
        "predicted_power_kw",
        "actual_down_g0.05",
        "actual_up_g0.05",
        "pred_down_prob_g0.05",
        "pred_up_prob_g0.05",
    ]
    df = _prediction_frames(usecols)
    selector_model = "JumpRS"
    base = df[df["model"] == selector_model].copy()
    base["event_count"] = base["actual_down_g0.05"] + base["actual_up_g0.05"]
    sample_stats = base.groupby(["site_id", "sample_id"]).agg(
        event_count=("event_count", "sum"),
        down_count=("actual_down_g0.05", "sum"),
        up_count=("actual_up_g0.05", "sum"),
        power_range=("actual_power_kw", lambda s: float(s.max() - s.min())),
        rmse=("predicted_power_kw", lambda s: 0.0),
    ).reset_index()
    errs = base.assign(se=(base["predicted_power_kw"] - base["actual_power_kw"]) ** 2)
    rmse = errs.groupby(["site_id", "sample_id"])["se"].mean().pow(0.5).reset_index(name="jump_rs_rmse")
    sample_stats = sample_stats.drop(columns=["rmse"]).merge(rmse, on=["site_id", "sample_id"])
    picks = []
    for site_id, site_stats in sample_stats.groupby("site_id"):
        # Prefer (1) non-trivial ramp activity, (2) good JumpRS fit quality, to avoid
        # unintentionally selecting extremely hard/volatile samples that make every model look bad.
        moderate = site_stats[
            (site_stats["down_count"] > 0)
            & (site_stats["up_count"] > 0)
            & (site_stats["event_count"] >= 2)
            & (site_stats["event_count"] <= 12)
        ].copy()
        if moderate.empty:
            moderate = site_stats[(site_stats["event_count"] >= 2) & (site_stats["event_count"] <= 8)].copy()
        if moderate.empty:
            moderate = site_stats[(site_stats["event_count"] > 0) & (site_stats["event_count"] <= 12)].copy()
        if moderate.empty:
            moderate = site_stats.copy()
        cutoff = moderate["jump_rs_rmse"].quantile(0.25)
        moderate = moderate[moderate["jump_rs_rmse"] <= cutoff]
        pick = moderate.sort_values(["event_count", "power_range"], ascending=[False, False]).iloc[0]
        picks.append(pick)
    picks = pd.DataFrame(picks)
    models = ["JumpRS", "QuantileGRU", "MC Dropout", "PatchTST", "iTransformer", "TimesNet", "Persistence", "TimeDiff-style", "NsDiff-style"]

    fig, axes = plt.subplots(
        len(picks) * 3,
        1,
        figsize=(7.1, 2.15 * len(picks) * 1.05),
        sharex=False,
        gridspec_kw={"height_ratios": [2.2, 1.0, 1.0] * len(picks)},
    )
    axes = np.atleast_1d(axes)

    for row_idx, (_, pick) in enumerate(picks.iterrows()):
        ax_pow = axes[row_idx * 3 + 0]
        ax_down = axes[row_idx * 3 + 1]
        ax_up = axes[row_idx * 3 + 2]

        sub = df[
            (df["site_id"] == pick["site_id"])
            & (df["sample_id"] == pick["sample_id"])
            & (df["model"].isin(models))
        ]
        ref = sub[sub["model"] == selector_model].sort_values("horizon_step")
        x = ref["horizon_step"].to_numpy() * 0.25

        # Power trajectories
        ax_pow.plot(x, ref["actual_power_kw"], color="#000000", lw=1.6, label="Measured")
        ax_pow.plot(x, ref["p_cs_next_kw"], color="#666666", lw=1.1, ls="--", label="Clear-sky")
        for model in models:
            s = sub[sub["model"] == model].sort_values("horizon_step")
            if s.empty:
                continue
            lw = 1.25 if model == selector_model else 0.95
            ax_pow.plot(x, s["predicted_power_kw"], lw=lw, label=model, color=COLORS.get(model))

        # Ramp-event indicators (direction-specific), drawn as vlines to avoid color confusion with model curves.
        down_evt = ref["actual_down_g0.05"].to_numpy().astype(bool)
        up_evt = ref["actual_up_g0.05"].to_numpy().astype(bool)
        y0 = float(ref["actual_power_kw"].min())
        y1 = y0 + 0.06 * float(max(ref["actual_power_kw"].max() - y0, 1.0))
        if down_evt.any():
            ax_pow.vlines(x[down_evt], y0, y1, colors="#0072B2", lw=1.2, alpha=0.9, label="Down-ramp (actual)")
        if up_evt.any():
            ax_pow.vlines(x[up_evt], y0, y1, colors="#E69F00", lw=1.2, alpha=0.9, label="Up-ramp (actual)")

        ax_pow.set_title(f"Case Study: {pick['site_id']} (6-h horizon, $\\gamma=0.05$)")
        ax_pow.set_ylabel("Power (kW)")

        # Event probability forecasts (focus on the main model to avoid clutter).
        main = sub[sub["model"] == selector_model].sort_values("horizon_step")
        ax_down.plot(x, main["pred_down_prob_g0.05"], color="#0072B2", lw=1.4, label="Pred down-ramp prob")
        ax_up.plot(x, main["pred_up_prob_g0.05"], color="#E69F00", lw=1.4, label="Pred up-ramp prob")
        ax_down.scatter(x[down_evt], main["pred_down_prob_g0.05"].to_numpy()[down_evt], s=16, color="#0072B2", zorder=5, marker="s", label="Down-ramp (actual)")
        ax_up.scatter(x[up_evt], main["pred_up_prob_g0.05"].to_numpy()[up_evt], s=16, color="#E69F00", zorder=5, marker="s", label="Up-ramp (actual)")
        for ax, title in [(ax_down, "Down-ramp Event Probability"), (ax_up, "Up-ramp Event Probability")]:
            ax.set_ylim(0, 1)
            ax.set_ylabel("Prob.")
            ax.set_title(title)
            ax.grid(True, alpha=0.25)
        ax_up.set_xlabel("Lead time (h)")

    axes[0].legend(frameon=False, ncol=4, fontsize=6, loc="best")
    fig.tight_layout()
    fig.savefig(FIG / "case_study_multisite.pdf")
    plt.close(fig)


def main():
    _setup()
    plot_trajectory_leaderboard()
    plot_event_bar_gamma005()
    plot_event_horizon()
    plot_trajectory_horizon()
    plot_crps_selected_horizons()
    plot_case_study()
    print(f"Wrote paper figures to {FIG}")


if __name__ == "__main__":
    main()
