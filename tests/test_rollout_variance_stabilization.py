import copy

import torch

from src.models.jumprs import rollout_marginal_moments
from src.training.losses import jumprs_loss


def _params(batch, horizon):
    base = torch.full((batch, horizon), 0.1)
    return {
        "kappa": base,
        "xbar": torch.full_like(base, 0.6),
        "sigma": torch.full_like(base, 0.2),
        "lambda": base,
        "mu_j": torch.zeros_like(base),
        "eta_j": torch.full_like(base, 0.05),
    }


def _cfg(scale=1.0, distribution="conditional_mixture"):
    return {
        "model": {
            "truncation_order": 2,
            "dt": 1.0,
            "eps_var": 1e-3,
            "x_min": 0.0,
            "x_max": 1.2,
            "rollout_variance_scale": scale,
            "crps_distribution": distribution,
            "recursive_crps_weight": 0.5,
            "conditioning_path_mode": "mean",
            "crps_scale": "power",
            "power_loss_scale": "global_max_capacity",
            "transition_nll_weight": 1.0,
            "mixture_crps_weight": 1.0,
            "adaptive_loss_balance": False,
        }
    }


def test_rollout_variance_scale_reduces_recursive_uncertainty():
    params = _params(3, 4)
    z = torch.full((3,), 0.5)
    _, std_full = rollout_marginal_moments(z, params, _cfg(1.0))
    _, std_stable = rollout_marginal_moments(z, params, _cfg(0.5))
    assert torch.all(std_stable < std_full)


def test_hybrid_rollout_keeps_exactly_two_loss_parts():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.param_head = torch.nn.Linear(1, 1)
            self.register_buffer("loss_ema_nll", torch.tensor(1.0))
            self.register_buffer("loss_ema_crps", torch.tensor(1.0))

    batch_size, horizon = 3, 4
    batch = {
        "X_hist": torch.full((batch_size, 2, 1), 0.5),
        "y_x": torch.full((batch_size, horizon), 0.55),
        "p_cs_next": torch.full((batch_size, horizon), 100.0),
        "capacity_kw": torch.full((batch_size, horizon), 100.0),
    }
    model_cfg = _cfg(0.7, "hybrid_rollout")
    loss, parts = jumprs_loss(
        Model(),
        batch,
        _params(batch_size, horizon),
        {"sites": [{"capacity_kw": 100.0}]},
        model_cfg,
    )
    assert torch.isfinite(loss)
    assert set(parts) == {"transition_nll", "mixture_crps"}
