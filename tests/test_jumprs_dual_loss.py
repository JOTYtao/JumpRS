import torch

from src.models.jumprs import JumpRS
from src.training.losses import jumprs_loss


def test_jumprs_loss_contains_only_transition_nll_and_mixture_crps():
    model = JumpRS(
        input_dim=3,
        history_steps=4,
        forecast_steps=2,
        hidden_dim=16,
        num_layers=1,
        nhead=4,
        num_regimes=2,
    )
    batch = {
        "X_hist": torch.rand(3, 4, 3),
        "y_x": torch.rand(3, 2),
        "y_power": torch.rand(3, 2),
        "p_cs_next": torch.full((3, 2), 10.0),
        "capacity_kw": torch.full((3, 2), 10.0),
    }
    model_cfg = {
        "model": {
            "truncation_order": 2,
            "dt": 1.0,
            "eps_var": 1.0e-3,
            "x_min": 0.0,
            "x_max": 1.2,
            "transition_nll_weight": 1.0,
            "mixture_crps_weight": 1.0,
            "crps_scale": "power",
            "power_loss_scale": "per_site_capacity",
        }
    }
    params = model(batch["X_hist"], future_clear_sky=torch.ones(3, 2))
    loss, parts = jumprs_loss(model, batch, params, {"sites": []}, model_cfg)
    loss.backward()

    assert set(parts) == {"transition_nll", "mixture_crps"}
    assert torch.isfinite(loss)
