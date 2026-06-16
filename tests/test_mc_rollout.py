import torch

from src.training.train import _mc_rollout_predictions


def test_mc_rollout_returns_samples_by_member_window_and_horizon():
    windows, horizon = 3, 4
    base = torch.full((windows, horizon), 0.1)
    params = {
        "kappa": base,
        "xbar": torch.full_like(base, 0.6),
        "sigma": torch.full_like(base, 0.05),
        "lambda": base,
        "mu_j": torch.zeros_like(base),
        "eta_j": torch.full_like(base, 0.02),
    }
    batch = {
        "X_hist": torch.full((windows, 2, 1), 0.5),
        "p_prev": torch.full((windows, horizon), 50.0),
        "p_cs_next": torch.full((windows, horizon), 100.0),
        "capacity_kw": torch.full((windows, horizon), 100.0),
    }
    data_cfg = {"ramp": {"thresholds": [0.05]}, "site": {"capacity_kw": 100.0}}
    model_cfg = {
        "model": {
            "mc_event_samples": 7,
            "mc_event_seed": 42,
            "x_min": 0.0,
            "x_max": 1.2,
            "truncation_order": 2,
            "dt": 1.0,
            "eps_var": 1e-3,
        }
    }

    down, up, sample_x = _mc_rollout_predictions(
        params, batch, data_cfg, model_cfg, torch.device("cpu")
    )

    assert down.shape == (windows, horizon, 1)
    assert up.shape == (windows, horizon, 1)
    assert sample_x.shape == (7, windows, horizon)
