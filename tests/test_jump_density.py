import torch
from src.models.jumprs import transition_components, one_step_ramp_probabilities


def test_transition_shapes_with_synthetic_unit_tensors_only():
    cfg = {"model": {"truncation_order": 3, "dt": 1.0, "eps_var": 1e-3, "x_min": 0.0, "x_max": 1.2}}
    z = torch.tensor([0.5, 0.7])
    params = {"kappa": torch.ones(2), "xbar": torch.ones(2)*0.6, "sigma": torch.ones(2)*0.1, "lambda": torch.ones(2)*0.2, "mu_j": torch.zeros(2), "eta_j": torch.ones(2)*0.05}
    log_w, mean, std = transition_components(params, z, cfg)
    assert log_w.shape == (2, 4)
    assert mean.shape == (2, 4)
    assert torch.all(std > 0)
    down, up = one_step_ramp_probabilities(z, torch.ones(2)*10, torch.ones(2)*10, 10.0, [0.05, 0.10], params, cfg)
    assert down.shape == (2, 2)
    assert up.shape == (2, 2)
