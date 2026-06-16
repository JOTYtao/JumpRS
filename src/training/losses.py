import math

import torch

from src.models.jumprs import (
    conditioning_path,
    observed_conditioning_path,
    rollout_marginal_moments,
    transition_components,
    transition_log_prob,
)


def _capacity_scale(batch, data_cfg=None, model_cfg=None):
    capacity = batch.get("capacity_kw")
    if capacity is None:
        return None
    if model_cfg is not None and model_cfg["model"].get("power_loss_scale", "per_site_capacity") == "global_max_capacity":
        capacities = [float(site["capacity_kw"]) for site in (data_cfg or {}).get("sites", [])]
        scale = max(capacities) if capacities else float(capacity.max().detach())
        return torch.full_like(capacity, max(scale, 1.0))
    while capacity.ndim < 2:
        capacity = capacity.unsqueeze(-1)
    return torch.clamp(capacity, min=1.0)


def _normal_absolute_moment(delta, scale):
    scale = torch.clamp(scale, min=1e-6)
    z = delta / scale
    pdf = torch.exp(-0.5 * z.pow(2)) / math.sqrt(2.0 * math.pi)
    cdf = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    return 2.0 * scale * pdf + delta * (2.0 * cdf - 1.0)


def gaussian_mixture_crps(target, log_weights, means, stds, reduction="mean"):
    """Closed-form CRPS for the finite Gaussian mixture induced by JumpRS."""
    weights = torch.exp(log_weights)
    observation_term = (
        weights * _normal_absolute_moment(target.unsqueeze(-1) - means, stds)
    ).sum(dim=-1)
    pair_delta = means.unsqueeze(-1) - means.unsqueeze(-2)
    pair_scale = torch.sqrt(stds.unsqueeze(-1).pow(2) + stds.unsqueeze(-2).pow(2))
    pair_weight = weights.unsqueeze(-1) * weights.unsqueeze(-2)
    mixture_term = 0.5 * (
        pair_weight * _normal_absolute_moment(pair_delta, pair_scale)
    ).sum(dim=(-1, -2))
    values = observation_term - mixture_term
    return values.mean() if reduction == "mean" else values


def gaussian_crps(target, mean, std):
    """Closed-form CRPS for a Gaussian predictive marginal."""
    return _normal_absolute_moment(target - mean, std) - std / math.sqrt(math.pi)


def bounded_mixture_nll(target, log_weights, means, stds, x_min, x_max, boundary_tol=1e-5):
    """NLL of the Gaussian mixture after projection onto the physical interval."""
    normal = torch.distributions.Normal(
        torch.tensor(0.0, device=target.device, dtype=target.dtype),
        torch.tensor(1.0, device=target.device, dtype=target.dtype),
    )
    lower_log_mass = torch.logsumexp(
        log_weights + torch.log(torch.clamp(normal.cdf((x_min - means) / stds), min=1e-12)),
        dim=-1,
    )
    upper_log_mass = torch.logsumexp(
        log_weights
        + torch.log(torch.clamp(1.0 - normal.cdf((x_max - means) / stds), min=1e-12)),
        dim=-1,
    )
    log_density = torch.logsumexp(
        log_weights
        - 0.5 * math.log(2.0 * math.pi)
        - torch.log(stds)
        - 0.5 * ((target.unsqueeze(-1) - means) / stds).pow(2),
        dim=-1,
    )
    log_prob = torch.where(
        target <= x_min + boundary_tol,
        lower_log_mass,
        torch.where(target >= x_max - boundary_tol, upper_log_mass, log_density),
    )
    return -log_prob


def bounded_mixture_crps(target, log_weights, means, stds, x_min, x_max, grid_size=33):
    """Differentiable quadrature CRPS for the projected bounded mixture."""
    weights = torch.exp(log_weights)
    grid = torch.linspace(
        float(x_min),
        float(x_max),
        int(grid_size),
        device=target.device,
        dtype=target.dtype,
    )
    normal = torch.distributions.Normal(
        torch.tensor(0.0, device=target.device, dtype=target.dtype),
        torch.tensor(1.0, device=target.device, dtype=target.dtype),
    )
    cdf = (
        weights.unsqueeze(-2)
        * normal.cdf((grid.view(*([1] * target.ndim), -1, 1) - means.unsqueeze(-2)) / stds.unsqueeze(-2))
    ).sum(dim=-1)
    observed_cdf = (grid.view(*([1] * target.ndim), -1) >= target.unsqueeze(-1)).to(target.dtype)
    return torch.trapezoid((cdf - observed_cdf).pow(2), grid, dim=-1)


def _horizon_weights(values, power):
    if float(power) == 0.0:
        return torch.ones_like(values)
    steps = torch.arange(1, values.shape[1] + 1, device=values.device, dtype=values.dtype)
    weights = steps.pow(float(power))
    weights = weights / weights.mean()
    return weights.unsqueeze(0).expand_as(values)


def _site_focus_weights(batch, values, cfg):
    focus_capacity = cfg.get("site_focus_capacity_kw")
    if focus_capacity is None:
        return torch.ones_like(values)
    capacity = batch.get("capacity_kw")
    if capacity is None:
        raise ValueError("capacity_kw is required for site-focused optimization.")
    if capacity.ndim > 1:
        capacity = capacity[:, 0]
    focused = torch.isclose(
        capacity,
        torch.tensor(float(focus_capacity), device=capacity.device, dtype=capacity.dtype),
        rtol=0.0,
        atol=1e-4,
    )
    sample_weights = torch.where(
        focused,
        torch.full_like(capacity, float(cfg.get("site_focus_weight", 1.0))),
        torch.ones_like(capacity),
    )
    sample_weights = sample_weights / sample_weights.mean()
    return sample_weights.unsqueeze(-1).expand_as(values)


def _adaptive_loss_weights(model, transition_nll, mixture_crps, cfg):
    if not bool(cfg.get("adaptive_loss_balance", False)):
        return (
            float(cfg.get("transition_nll_weight", 1.0)),
            float(cfg.get("mixture_crps_weight", 1.0)),
        )
    decay = float(cfg.get("adaptive_loss_ema_decay", 0.95))
    if model.training:
        if cfg.get("adaptive_loss_mode", "magnitude") == "gradient":
            nll_signal = torch.autograd.grad(
                transition_nll,
                model.param_head.weight,
                retain_graph=True,
                allow_unused=False,
            )[0].detach().norm()
            crps_signal = torch.autograd.grad(
                mixture_crps,
                model.param_head.weight,
                retain_graph=True,
                allow_unused=False,
            )[0].detach().norm()
        else:
            nll_signal = transition_nll.detach().abs()
            crps_signal = mixture_crps.detach().abs()
        with torch.no_grad():
            model.loss_ema_nll.mul_(decay).add_((1.0 - decay) * nll_signal.clamp(min=1e-4))
            model.loss_ema_crps.mul_(decay).add_((1.0 - decay) * crps_signal.clamp(min=1e-4))
    inv = torch.stack(
        [
            1.0 / torch.clamp(model.loss_ema_nll, min=1e-4),
            1.0 / torch.clamp(model.loss_ema_crps, min=1e-4),
        ]
    )
    weights = 2.0 * inv / inv.sum()
    lower = float(cfg.get("adaptive_loss_min_weight", 0.25))
    upper = 2.0 - lower
    weights = torch.clamp(weights, min=lower, max=upper)
    weights = 2.0 * weights / weights.sum()
    strength = float(cfg.get("adaptive_loss_strength", 1.0))
    base = torch.tensor(
        [
            float(cfg.get("transition_nll_weight", 1.0)),
            float(cfg.get("mixture_crps_weight", 1.0)),
        ],
        device=weights.device,
        dtype=weights.dtype,
    )
    base = 2.0 * base / torch.clamp(base.sum(), min=1e-6)
    weights = (1.0 - strength) * base + strength * weights
    return weights[0], weights[1]


def jumprs_loss(model, batch, params, data_cfg, model_cfg):
    """Train JumpRS with exactly two proper distributional objectives."""
    cfg = model_cfg["model"]
    y_x = batch["y_x"]
    if y_x.ndim == 1:
        y_x = y_x.unsqueeze(1)

    z = batch["X_hist"][:, -1, 0]
    path_mode = cfg.get("conditioning_path_mode", "mean")
    if path_mode == "observed":
        z_paths = [observed_conditioning_path(z, y_x)]
    elif path_mode == "hybrid":
        z_paths = [observed_conditioning_path(z, y_x), conditioning_path(z, params, model_cfg)]
    else:
        z_paths = [conditioning_path(z, params, model_cfg)]

    nll_paths = []
    crps_paths = []
    for z_path in z_paths:
        log_weights, component_means, component_stds = transition_components(params, z_path, model_cfg)
        if bool(cfg.get("bounded_distribution", False)):
            nll_paths.append(
                bounded_mixture_nll(
                    y_x,
                    log_weights,
                    component_means,
                    component_stds,
                    float(cfg["x_min"]),
                    float(cfg["x_max"]),
                )
            )
            crps_paths.append(
                bounded_mixture_crps(
                    y_x,
                    log_weights,
                    component_means,
                    component_stds,
                    float(cfg["x_min"]),
                    float(cfg["x_max"]),
                    int(cfg.get("bounded_crps_grid_size", 33)),
                )
            )
        else:
            nll_paths.append(-transition_log_prob(y_x, z_path, params, model_cfg))
            crps_paths.append(
                gaussian_mixture_crps(
                    y_x,
                    log_weights,
                    component_means,
                    component_stds,
                    reduction="none",
                )
            )
    transition_nll_values = torch.stack(nll_paths).mean(dim=0)
    crps_distribution = cfg.get("crps_distribution", "conditional_mixture")
    if crps_distribution == "marginal_moment":
        marginal_mean, marginal_std = rollout_marginal_moments(z, params, model_cfg)
        mixture_crps = gaussian_crps(y_x, marginal_mean, marginal_std)
    elif crps_distribution == "hybrid_rollout":
        conditional_crps = torch.stack(crps_paths).mean(dim=0)
        marginal_mean, marginal_std = rollout_marginal_moments(z, params, model_cfg)
        recursive_crps = gaussian_crps(y_x, marginal_mean, marginal_std)
        recursive_weight = float(cfg.get("recursive_crps_weight", 0.5))
        mixture_crps = (
            (1.0 - recursive_weight) * conditional_crps
            + recursive_weight * recursive_crps
        )
    else:
        mixture_crps = torch.stack(crps_paths).mean(dim=0)
    horizon_weights = _horizon_weights(transition_nll_values, cfg.get("horizon_weight_power", 0.0))
    site_weights = _site_focus_weights(batch, transition_nll_values, cfg)
    transition_nll = (transition_nll_values * horizon_weights * site_weights).mean()
    objective_mode = str(cfg.get("loss_objective_mode", "transition_crps"))
    if objective_mode in {"filter_crps", "zakai_kl"}:
        if "_filter_nll" not in params or "_belief_kl" not in params:
            raise ValueError(f"{objective_mode} requires belief-filtered model outputs.")
        transition_nll = transition_nll + float(cfg.get("filter_nll_weight", 0.1)) * params[
            "_filter_nll"
        ].mean()
    if cfg.get("crps_scale", "power") == "power":
        capacity = _capacity_scale(batch, data_cfg, model_cfg)
        if capacity is None:
            raise ValueError("capacity_kw is required when crps_scale='power'.")
        mixture_crps = mixture_crps * batch["p_cs_next"] / capacity
    mixture_crps = (mixture_crps * horizon_weights * site_weights).mean()

    if objective_mode == "zakai_kl":
        second_objective = float(cfg.get("belief_kl_weight", 0.01)) * params["_belief_kl"].mean()
    else:
        second_objective = mixture_crps
    nll_weight, crps_weight = _adaptive_loss_weights(
        model, transition_nll, second_objective, cfg
    )
    total = nll_weight * transition_nll + crps_weight * second_objective
    return total, {
        "transition_nll": float(transition_nll.detach()),
        "mixture_crps": float(second_objective.detach()),
    }
