import torch

from src.models.jumprs import JumpRS
from src.training.losses import jumprs_loss
from src.training.train import _mc_rollout_predictions


def _model():
    return JumpRS(
        input_dim=3,
        history_steps=8,
        forecast_steps=4,
        hidden_dim=32,
        num_layers=1,
        nhead=4,
        num_regimes=3,
        belief_filtering=True,
        state_conditioned_decoder=True,
    )


def _cfg():
    return {
        "model": {
            "mc_event_samples": 7,
            "mc_event_seed": 42,
            "x_min": 0.0,
            "x_max": 1.2,
            "truncation_order": 2,
            "dt": 1.0,
            "eps_var": 1e-3,
            "rollout_variance_scale": 1.0,
        }
    }


def test_belief_filter_outputs_normalized_stepwise_regimes():
    model = _model()
    x = torch.rand(3, 8, 3)
    params = model(x, future_clear_sky=torch.rand(3, 4))
    assert params["pi"].shape == (3, 4, 3)
    assert torch.allclose(params["pi"].sum(dim=-1), torch.ones(3, 4), atol=1e-5)
    assert params["_rollout_belief0"].shape == (3, 3)


def test_dynamic_rollout_redecodes_from_sampled_state():
    model = _model()
    batch = {
        "X_hist": torch.rand(3, 8, 3),
        "p_prev": torch.full((3, 4), 50.0),
        "p_cs_next": torch.full((3, 4), 100.0),
        "capacity_kw": torch.full((3, 4), 100.0),
    }
    params = model(batch["X_hist"], future_clear_sky=torch.ones(3, 4))
    data_cfg = {"ramp": {"thresholds": [0.05]}, "site": {"capacity_kw": 100.0}}
    down, up, sample_x = _mc_rollout_predictions(
        params, batch, data_cfg, _cfg(), torch.device("cpu"), model=model
    )
    assert down.shape == (3, 4, 1)
    assert up.shape == (3, 4, 1)
    assert sample_x.shape == (7, 3, 4)
    assert torch.isfinite(sample_x).all()


def test_filter_likelihood_and_kl_support_two_loss_modes():
    model = _model()
    batch = {
        "X_hist": torch.rand(3, 8, 3),
        "y_x": torch.rand(3, 4),
        "p_cs_next": torch.full((3, 4), 100.0),
        "capacity_kw": torch.full((3, 4), 100.0),
    }
    params = model(batch["X_hist"], future_clear_sky=torch.ones(3, 4))
    base = {
        "truncation_order": 2,
        "dt": 1.0,
        "eps_var": 1e-3,
        "x_min": 0.0,
        "x_max": 1.2,
        "conditioning_path_mode": "hybrid",
        "crps_distribution": "hybrid_rollout",
        "recursive_crps_weight": 0.5,
        "crps_scale": "power",
        "power_loss_scale": "per_site_capacity",
        "adaptive_loss_balance": False,
    }
    for mode in ["filter_crps", "zakai_kl"]:
        cfg = {"model": {**base, "loss_objective_mode": mode}}
        loss, parts = jumprs_loss(model, batch, params, {"sites": []}, cfg)
        assert torch.isfinite(loss)
        assert set(parts) == {"transition_nll", "mixture_crps"}
