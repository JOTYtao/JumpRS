import torch

from src.models.jumprs import JumpRS, observed_conditioning_path
from src.training.losses import (
    _site_focus_weights,
    bounded_mixture_crps,
    bounded_mixture_nll,
    jumprs_loss,
)


def test_bounded_scores_are_finite_and_differentiable():
    target = torch.tensor([[0.0, 0.6, 1.2]])
    logits = torch.zeros(1, 3, 2, requires_grad=True)
    means = torch.tensor([[[-0.1, 0.2], [0.5, 0.8], [1.0, 1.4]]], requires_grad=True)
    stds = torch.full_like(means, 0.15, requires_grad=True)
    log_weights = torch.log_softmax(logits, dim=-1)
    loss = bounded_mixture_nll(target, log_weights, means, stds, 0.0, 1.2).mean()
    loss = loss + bounded_mixture_crps(target, log_weights, means, stds, 0.0, 1.2).mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert means.grad is not None


def test_observed_path_uses_previous_measured_state():
    z0 = torch.tensor([0.1, 0.2])
    observed = torch.tensor([[0.3, 0.4, 0.5], [0.6, 0.7, 0.8]])
    path = observed_conditioning_path(z0, observed)
    assert torch.allclose(path, torch.tensor([[0.1, 0.3, 0.4], [0.2, 0.6, 0.7]]))


def test_dedicated_markov_predictor_and_adaptive_loss_remain_two_term():
    model = JumpRS(
        input_dim=3,
        history_steps=8,
        forecast_steps=4,
        hidden_dim=32,
        num_layers=1,
        nhead=4,
        num_regimes=3,
        dedicated_predictor=True,
        site_conditioning=True,
        markov_regimes=True,
    )
    batch = {
        "X_hist": torch.rand(3, 8, 3),
        "y_x": torch.rand(3, 4) * 1.2,
        "y_power": torch.rand(3, 4),
        "p_cs_next": torch.full((3, 4), 10.0),
        "capacity_kw": torch.full((3, 4), 10.0),
    }
    cfg = {
        "model": {
            "truncation_order": 2,
            "dt": 1.0,
            "eps_var": 1.0e-3,
            "x_min": 0.0,
            "x_max": 1.2,
            "crps_scale": "power",
            "power_loss_scale": "per_site_capacity",
            "bounded_distribution": True,
            "conditioning_path_mode": "observed",
            "adaptive_loss_balance": True,
            "adaptive_loss_strength": 0.5,
        }
    }
    params = model(
        batch["X_hist"],
        future_clear_sky=torch.ones(3, 4),
        site_context=torch.ones(3),
    )
    loss, parts = jumprs_loss(model, batch, params, {"sites": []}, cfg)
    loss.backward()
    assert set(parts) == {"transition_nll", "mixture_crps"}
    assert torch.allclose(params["pi"].sum(dim=-1), torch.ones(3, 4), atol=1e-5)
    assert torch.isfinite(loss)


def test_site_focus_weights_only_upweight_the_configured_capacity():
    values = torch.ones(3, 2)
    batch = {"capacity_kw": torch.tensor([[1.0, 1.0], [146.64, 146.64], [1.12, 1.12]])}
    weights = _site_focus_weights(
        batch,
        values,
        {"site_focus_capacity_kw": 146.64, "site_focus_weight": 4.0},
    )
    assert weights[1, 0] > weights[0, 0]
    assert torch.isclose(weights.mean(), torch.tensor(1.0))
