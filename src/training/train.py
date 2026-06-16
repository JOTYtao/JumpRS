import random
import numpy as np
import torch
import copy
from torch.utils.data import DataLoader, Dataset
from src.models.jumprs import (
    JumpRS,
    conditioning_path,
    one_step_ramp_probabilities,
    rollout_mean_centers,
    rollout_marginal_moments,
    transition_components,
)
from src.training.losses import jumprs_loss


class WindowDataset(Dataset):
    def __init__(self, split):
        self.split = split
    def __len__(self):
        return len(self.split["y_x"])
    def __getitem__(self, idx):
        out = {}
        for k, v in self.split.items():
            if k in {"time", "site_id"}:
                continue
            dtype = torch.bool if k == "ramp_valid" else torch.float32
            out[k] = torch.tensor(v[idx], dtype=dtype)
        return out


def choose_device(train_cfg):
    requested = train_cfg["training"].get("device", "auto")
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def _early_stopping_params(train_cfg):
    cfg = train_cfg["training"]
    return int(cfg.get("patience", 10)), float(cfg.get("min_delta", 1e-4))


def _make_jumprs_model(splits, data_cfg, model_cfg, encoder_type):
    kwargs = dict(
        input_dim=splits["train"]["X_hist"].shape[-1],
        forecast_steps=int(data_cfg["window"]["forecast_steps"]),
        hidden_dim=int(model_cfg["model"]["hidden_dim"]),
        num_layers=int(model_cfg["model"]["num_layers"]),
        dropout=float(model_cfg["model"]["dropout"]),
        num_regimes=int(model_cfg["model"]["num_regimes"]),
        x_min=float(model_cfg["model"]["x_min"]),
        x_max=float(model_cfg["model"]["x_max"]),
        m_max=float(model_cfg["model"]["m_max"]),
        eps=float(model_cfg["model"]["eps"]),
        quantiles=tuple(float(q) for q in model_cfg["model"].get("quantiles", [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])),
        quantile_residual_scale=float(model_cfg["model"].get("quantile_residual_scale", 0.35)),
        point_residual_scale=float(model_cfg["model"].get("point_residual_scale", 0.35)),
        use_forecast_heads=bool(model_cfg["model"].get("use_forecast_heads", False)),
        forecast_head_anchor=str(model_cfg["model"].get("forecast_head_anchor", "persistence")),
        forecast_head_arch=str(model_cfg["model"].get("forecast_head_arch", "attention")),
        dedicated_predictor=bool(model_cfg["model"].get("dedicated_predictor", False)),
        site_conditioning=bool(model_cfg["model"].get("site_conditioning", False)),
        markov_regimes=bool(model_cfg["model"].get("markov_regimes", False)),
        belief_filtering=bool(model_cfg["model"].get("belief_filtering", False)),
        state_conditioned_decoder=bool(model_cfg["model"].get("state_conditioned_decoder", False)),
    )
    if encoder_type != "JumpRS":
        raise ValueError(
            "Only the main JumpRS model is retained; set encoder_type to JumpRS."
        )
    return JumpRS(
        history_steps=splits["train"]["X_hist"].shape[1],
        **kwargs,
    )


def _attach_input_scaler(model, split, device):
    x = split["X_hist"].astype("float32")
    mean = torch.tensor(x.mean(axis=(0, 1), keepdims=True), dtype=torch.float32, device=device)
    std = torch.tensor(x.std(axis=(0, 1), keepdims=True) + 1e-6, dtype=torch.float32, device=device)
    model.register_buffer("_x_mean", mean)
    model.register_buffer("_x_std", std)


def _model_input(model, x_hist):
    if hasattr(model, "_x_mean") and hasattr(model, "_x_std"):
        return (x_hist - model._x_mean) / model._x_std
    return x_hist


def _forward_model(model, batch):
    x_hist = _model_input(model, batch["X_hist"])
    if getattr(model, "uses_future_context", False):
        capacity = torch.clamp(batch["capacity_kw"], min=1e-6)
        future_clear_sky = torch.clamp(batch["p_cs_next"] / capacity, 0.0, 1.5)
        site_context = torch.log1p(capacity[:, 0]) / 5.0
        return model(x_hist, future_clear_sky=future_clear_sky, site_context=site_context)
    return model(x_hist)


def _evaluate_jumprs(model, loader, data_cfg, model_cfg, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            params = _forward_model(model, batch)
            loss, _ = jumprs_loss(model, batch, params, data_cfg, model_cfg)
            losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def train_jumprs(splits, data_cfg, model_cfg, train_cfg, encoder_type=None, model_name=None):
    seed = int(train_cfg["training"]["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = choose_device(train_cfg)
    encoder_type = encoder_type or model_cfg["model"].get("encoder_type", "JumpRS")
    model_name = model_name or "JumpRS"
    model = _make_jumprs_model(splits, data_cfg, model_cfg, encoder_type).to(device)
    if bool(model_cfg["model"].get("use_input_scaler", False)):
        _attach_input_scaler(model, splits["train"], device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["training"]["learning_rate"]), weight_decay=float(train_cfg["training"]["weight_decay"]))
    loader = DataLoader(WindowDataset(splits["train"]), batch_size=int(train_cfg["training"]["batch_size"]), shuffle=True, num_workers=int(train_cfg["training"]["num_workers"]))
    val_loader = DataLoader(WindowDataset(splits["validation"]), batch_size=int(train_cfg["training"]["batch_size"]), shuffle=False, num_workers=int(train_cfg["training"]["num_workers"]))
    patience, min_delta = _early_stopping_params(train_cfg)
    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    history = []
    for epoch in range(int(train_cfg["training"]["max_epochs"])):
        model.train()
        losses = []
        part_rows = []
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad()
            params = _forward_model(model, batch)
            loss, parts = jumprs_loss(model, batch, params, data_cfg, model_cfg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["training"]["gradient_clip"]))
            opt.step()
            losses.append(float(loss.detach().cpu()))
            part_rows.append(parts)
        train_loss = float(np.mean(losses))
        val_loss = _evaluate_jumprs(model, val_loader, data_cfg, model_cfg, device)
        improved = val_loss < best_val - min_delta
        if improved:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
        part_means = {
            f"train_{key}": float(np.mean([row[key] for row in part_rows]))
            for key in part_rows[0]
        }
        history.append(
            {
                "model": model_name,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_val_loss": best_val,
                "bad_epochs": bad_epochs,
                **part_means,
            }
        )
        print(f"{model_name} epoch {epoch + 1}/{int(train_cfg['training']['max_epochs'])}: train={train_loss:.6f} val={val_loss:.6f}", flush=True)
        if bad_epochs >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, device


def _mc_rollout_predictions(params, batch, data_cfg, model_cfg, device, model=None):
    """Generate one coherent recursive rollout for trajectories and ramp risks."""
    cfg = model_cfg["model"]
    n_samples = int(cfg.get("mc_event_samples", 128))
    thresholds = [float(g) for g in data_cfg["ramp"]["thresholds"]]
    generator = torch.Generator(device=device)
    generator.manual_seed(int(cfg.get("mc_event_seed", 42)))
    z = batch["X_hist"][:, -1, 0].unsqueeze(0).expand(n_samples, -1)
    p_prev = batch["p_prev"]
    if p_prev.ndim == 2:
        prev_power = p_prev[:, 0].unsqueeze(0).expand(n_samples, -1)
    else:
        prev_power = p_prev.unsqueeze(0).expand(n_samples, -1)
    capacity = batch.get("capacity_kw")
    if capacity is None:
        capacity = torch.full_like(batch["p_cs_next"], float(data_cfg["site"]["capacity_kw"]))
    down_steps, up_steps, x_steps = [], [], []
    x_min = float(cfg["x_min"])
    x_max = float(cfg["x_max"])
    dynamic_belief = (
        model is not None
        and getattr(model, "belief_filtering", False)
        and "_rollout_h" in params
        and "_rollout_belief0" in params
    )
    if dynamic_belief:
        belief = params["_rollout_belief0"].unsqueeze(0).expand(n_samples, -1, -1)
    for tau in range(params["kappa"].shape[1]):
        step_params = {}
        if dynamic_belief:
            h_tau = params["_rollout_h"][:, tau].unsqueeze(0).expand(n_samples, -1, -1)
            flat_step, flat_belief = model._decode_belief_step(
                h_tau.reshape(-1, h_tau.shape[-1]),
                z.reshape(-1),
                belief.reshape(-1, belief.shape[-1]),
            )
            step_params = {
                key: value.reshape(n_samples, z.shape[1], *value.shape[1:])
                for key, value in flat_step.items()
            }
            belief = flat_belief.reshape(n_samples, z.shape[1], -1)
        elif "regime_lambda_down" in params:
            param_keys = [
                "pi",
                "kappa",
                "xbar",
                "sigma",
                "lambda_down",
                "lambda_up",
                "mu_down",
                "mu_up",
                "eta_down",
                "eta_up",
                "regime_kappa",
                "regime_xbar",
                "regime_sigma",
                "regime_lambda_down",
                "regime_lambda_up",
                "regime_mu_down",
                "regime_mu_up",
                "regime_eta_down",
                "regime_eta_up",
            ]
        elif "regime_kappa" in params:
            param_keys = [
                "pi",
                "kappa",
                "xbar",
                "sigma",
                "lambda",
                "mu_j",
                "eta_j",
                "regime_kappa",
                "regime_xbar",
                "regime_sigma",
                "regime_lambda",
                "regime_mu_j",
                "regime_eta_j",
            ]
        elif "lambda_down" in params and "lambda_up" in params:
            param_keys = ["kappa", "xbar", "sigma", "lambda_down", "lambda_up", "mu_down", "mu_up", "eta_down", "eta_up"]
        else:
            param_keys = ["kappa", "xbar", "sigma", "lambda", "mu_j", "eta_j"]
        if not dynamic_belief:
            for key in param_keys:
                value = params[key][:, tau].unsqueeze(0)
                step_params[key] = value.expand(n_samples, *([-1] * (value.ndim - 1)))
        log_w, comp_mean, comp_std = transition_components(step_params, z, model_cfg)
        weights = torch.exp(log_w)
        conditional_mean = (weights * comp_mean).sum(dim=-1)
        flat_idx = torch.distributions.Categorical(probs=weights.reshape(-1, weights.shape[-1])).sample()
        flat_idx = flat_idx.view(n_samples, -1, 1)
        mean_s = torch.gather(comp_mean, -1, flat_idx).squeeze(-1)
        std_s = torch.gather(comp_std, -1, flat_idx).squeeze(-1)
        eps = torch.randn(mean_s.shape, device=device, generator=generator)
        sampled_z = mean_s + std_s * eps
        variance_scale = float(cfg.get("rollout_variance_scale", 1.0))
        z = torch.clamp(
            conditional_mean + variance_scale * (sampled_z - conditional_mean),
            x_min,
            x_max,
        )
        power = z * batch["p_cs_next"][:, tau].unsqueeze(0)
        cap_tau = capacity[:, tau].unsqueeze(0) if capacity.ndim == 2 else capacity.unsqueeze(0)
        ramp = (power - prev_power) / torch.clamp(cap_tau, min=1.0)
        down_steps.append(torch.stack([(ramp <= -gamma).float().mean(dim=0) for gamma in thresholds], dim=-1))
        up_steps.append(torch.stack([(ramp >= gamma).float().mean(dim=0) for gamma in thresholds], dim=-1))
        x_steps.append(z)
        prev_power = power
    return (
        torch.stack(down_steps, dim=1),
        torch.stack(up_steps, dim=1),
        torch.stack(x_steps, dim=2),
    )


def predict_jumprs(model, split, data_cfg, model_cfg, device):
    model.eval()
    batch = {k: torch.tensor(v, dtype=torch.float32, device=device) for k, v in split.items() if k not in {"time", "site_id"}}
    with torch.no_grad():
        params = _forward_model(model, batch)
        z = batch["X_hist"][:, -1, 0]
        process_mean_x = rollout_mean_centers(z, params, model_cfg)
        marginal_mean_x, marginal_std_x = rollout_marginal_moments(z, params, model_cfg)
        use_marginal_prediction = bool(
            model_cfg["model"].get("use_marginal_moments_for_predictions", False)
        )
        if bool(model_cfg["model"].get("use_quantile_median_as_point", False)) and "point_quantile_x" in params:
            quantiles = tuple(float(q) for q in model_cfg["model"].get("quantiles", [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]))
            median_idx = quantiles.index(0.5) if 0.5 in quantiles else len(quantiles) // 2
            mean_x = params["point_quantile_x"][:, :, median_idx]
        else:
            mean_x = params.get(
                "point_mean_x",
                marginal_mean_x if use_marginal_prediction else process_mean_x,
            )
        z_path = conditioning_path(z, params, model_cfg)
        capacity = batch.get("capacity_kw")
        if capacity is None:
            capacity = torch.full_like(batch["p_cs_next"], float(data_cfg["site"]["capacity_kw"]))
        down, up = one_step_ramp_probabilities(z_path, batch["p_cs_prev"], batch["p_cs_next"], capacity, data_cfg["ramp"]["thresholds"], params, model_cfg)
        kernel_down, kernel_up = down, up
        mc_sample_x = None
        if model_cfg["model"].get("event_inference", "analytic") == "mc_path":
            mc_down, mc_up, mc_sample_x = _mc_rollout_predictions(
                params, batch, data_cfg, model_cfg, device, model=model
            )
            down, up = mc_down, mc_up
        if "aux_down_logit" in params and "aux_up_logit" in params:
            w = float(model_cfg["model"].get("aux_fusion_weight", 0.5))
            aux_down = torch.sigmoid(params["aux_down_logit"])
            aux_up = torch.sigmoid(params["aux_up_logit"])
            down = torch.clamp((1.0 - w) * down + w * aux_down, 1e-6, 1 - 1e-6)
            up = torch.clamp((1.0 - w) * up + w * aux_up, 1e-6, 1 - 1e-6)
        n_samples = 100
        generator = torch.Generator(device=device)
        generator.manual_seed(42)
        if use_marginal_prediction:
            eps = torch.randn(
                (n_samples, *marginal_mean_x.shape), device=device, generator=generator
            )
            sample_x = torch.clamp(
                marginal_mean_x.unsqueeze(0) + marginal_std_x.unsqueeze(0) * eps,
                float(model_cfg["model"]["x_min"]),
                float(model_cfg["model"]["x_max"]),
            )
        else:
            log_w, comp_mean, comp_std = transition_components(params, z_path, model_cfg)
            weights = torch.exp(log_w)
            comp_idx = torch.distributions.Categorical(probs=weights).sample((n_samples,))
            gather_idx = comp_idx.unsqueeze(-1)
            mean_s = torch.gather(comp_mean.unsqueeze(0).expand(n_samples, -1, -1, -1), -1, gather_idx).squeeze(-1)
            std_s = torch.gather(comp_std.unsqueeze(0).expand(n_samples, -1, -1, -1), -1, gather_idx).squeeze(-1)
            eps = torch.randn(mean_s.shape, device=device, generator=generator)
            sample_x = torch.clamp(mean_s + std_s * eps, float(model_cfg["model"]["x_min"]), float(model_cfg["model"]["x_max"]))
        if mc_sample_x is not None and bool(model_cfg["model"].get("use_mc_samples_for_intervals", False)):
            sample_x = mc_sample_x
        if "point_mean_x" in params:
            sample_x = torch.clamp(
                mean_x.unsqueeze(0) + sample_x - sample_x.mean(dim=0, keepdim=True),
                float(model_cfg["model"]["x_min"]),
                float(model_cfg["model"]["x_max"]),
            )
        samples_power = sample_x * batch["p_cs_next"].unsqueeze(0)
    out = {
        "mean_x": mean_x.cpu().numpy(),
        "power_mean": (mean_x * batch["p_cs_next"]).cpu().numpy(),
        "down_prob": down.cpu().numpy(),
        "up_prob": up.cpu().numpy(),
        "kernel_down_prob": kernel_down.cpu().numpy(),
        "kernel_up_prob": kernel_up.cpu().numpy(),
        "samples_power": samples_power.cpu().numpy(),
        "params": {k: v.cpu().numpy() for k, v in params.items() if not k.startswith("_")},
    }
    if "point_quantile_x" in params:
        out["quantiles"] = tuple(float(q) for q in model_cfg["model"].get("quantiles", [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]))
        out["quantile_power"] = (params["point_quantile_x"] * batch["p_cs_next"].unsqueeze(-1)).cpu().numpy()
    return out
