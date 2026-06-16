import torch

from src.models.jumprs import JumpRS


def test_jumprs_conditions_directional_kernel():
    model = JumpRS(
        input_dim=3,
        history_steps=16,
        forecast_steps=24,
        hidden_dim=32,
        num_layers=1,
        dropout=0.0,
        num_regimes=3,
    )
    params = model(torch.rand(2, 16, 3), future_clear_sky=torch.rand(2, 24))
    assert params["pi"].shape == (2, 24, 3)
    assert torch.all(params["regime_mu_down"] <= 0)
    assert torch.all(params["regime_mu_up"] >= 0)
