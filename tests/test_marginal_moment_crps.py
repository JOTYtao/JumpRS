import torch

from src.models.jumprs import rollout_marginal_moments
from src.training.losses import gaussian_crps


def test_gaussian_crps_and_marginal_moments_are_differentiable():
    batch, horizon = 2, 3
    z0 = torch.tensor([0.4, 0.7])
    base = torch.full((batch, horizon), 0.2, requires_grad=True)
    params = {
        "kappa": base + 0.1,
        "xbar": base + 0.4,
        "sigma": base + 0.05,
        "lambda_down": base,
        "lambda_up": base,
        "mu_down": -(base + 0.05),
        "mu_up": base + 0.05,
        "eta_down": base + 0.01,
        "eta_up": base + 0.01,
    }
    cfg = {
        "model": {
            "truncation_order": 1,
            "dt": 1.0,
            "eps_var": 1e-3,
            "x_min": 0.0,
            "x_max": 1.2,
        }
    }
    mean, std = rollout_marginal_moments(z0, params, cfg)
    loss = gaussian_crps(torch.full_like(mean, 0.5), mean, std).mean()
    loss.backward()
    assert mean.shape == (batch, horizon)
    assert std.shape == (batch, horizon)
    assert torch.isfinite(loss)
    assert base.grad is not None
