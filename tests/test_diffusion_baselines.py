import torch

from src.models.multistep_baselines import ConditionalDiffusionForecaster


def test_diffusion_forecasters_produce_multistep_samples():
    x = torch.rand(3, 16, 3)
    y = torch.rand(3, 24)
    for kind in ["timediff", "nsdiff"]:
        model = ConditionalDiffusionForecaster(3, 24, hidden_dim=32, diffusion_steps=4, kind=kind)
        loss = model.training_loss(x, y)
        samples = model.sample(x, n_samples=5)
        assert torch.isfinite(loss)
        assert samples.shape == (5, 3, 24)
        assert torch.all((samples >= 0.0) & (samples <= 1.2))
