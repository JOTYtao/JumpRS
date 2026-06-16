import torch

from src.training.losses import gaussian_mixture_crps


def test_gaussian_mixture_crps_is_finite_and_differentiable():
    target = torch.tensor([[0.2, 0.8]])
    logits = torch.tensor([[[0.0, -1.0], [-0.3, 0.0]]], requires_grad=True)
    means = torch.tensor([[[0.1, 0.4], [0.6, 0.9]]], requires_grad=True)
    stds = torch.full_like(means, 0.1, requires_grad=True)
    loss = gaussian_mixture_crps(target, torch.log_softmax(logits, dim=-1), means, stds)
    loss.backward()
    assert torch.isfinite(loss)
    assert means.grad is not None
