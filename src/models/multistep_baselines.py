import math
import copy

import numpy as np
import torch
from scipy.special import ndtr
from torch import nn
from torch.utils.data import DataLoader, Dataset


DEFAULT_QUANTILES = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)


def bounded_x(raw):
    return 1.2 * torch.sigmoid(raw)


def sorted_quantile_x(raw, horizon, quantiles):
    q = bounded_x(raw.view(-1, horizon, len(quantiles)))
    return torch.sort(q, dim=-1).values


class ForecastDataset(Dataset):
    def __init__(self, split):
        self.x = torch.tensor(split["X_hist"], dtype=torch.float32)
        self.y = torch.tensor(split["y_x"], dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class EncoderGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim=96, num_layers=2, dropout=0.1):
        super().__init__()
        self.encoder = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        _, h = self.encoder(x)
        return self.norm(h[-1])


class DeterministicSeqForecaster(nn.Module):
    def __init__(self, input_dim, horizon, hidden_dim=96, dropout=0.1, mc_dropout=False, quantiles=DEFAULT_QUANTILES):
        super().__init__()
        self.mc_dropout = mc_dropout
        self.quantiles = tuple(float(q) for q in quantiles)
        self.horizon = horizon
        self.encoder = EncoderGRU(input_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, horizon * len(self.quantiles)))

    def forward(self, x):
        return sorted_quantile_x(self.head(self.encoder(x)), self.horizon, self.quantiles)


class QuantileSeqForecaster(nn.Module):
    def __init__(self, input_dim, horizon, quantiles=DEFAULT_QUANTILES, hidden_dim=96, dropout=0.1):
        super().__init__()
        self.quantiles = tuple(float(q) for q in quantiles)
        self.horizon = horizon
        self.encoder = EncoderGRU(input_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, horizon * len(self.quantiles)))

    def forward(self, x):
        return sorted_quantile_x(self.head(self.encoder(x)), self.horizon, self.quantiles)


class PatchTSTForecaster(nn.Module):
    def __init__(self, input_dim, horizon, hidden_dim=96, patch_len=4, stride=2, dropout=0.1, nhead=4, num_layers=2, quantiles=DEFAULT_QUANTILES):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.input_dim = input_dim
        self.horizon = horizon
        self.quantiles = tuple(float(q) for q in quantiles)
        self.patch_proj = nn.Linear(input_dim * patch_len, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, horizon * len(self.quantiles)))

    def forward(self, x):
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        patches = patches.permute(0, 1, 3, 2).reshape(x.shape[0], -1, self.patch_len * self.input_dim)
        h = self.encoder(self.patch_proj(patches)).mean(dim=1)
        return sorted_quantile_x(self.head(h), self.horizon, self.quantiles)


class ITransformerForecaster(nn.Module):
    def __init__(self, input_dim, history_steps, horizon, hidden_dim=96, dropout=0.1, nhead=4, num_layers=2, quantiles=DEFAULT_QUANTILES):
        super().__init__()
        self.horizon = horizon
        self.quantiles = tuple(float(q) for q in quantiles)
        self.value_proj = nn.Linear(history_steps, hidden_dim)
        self.var_embed = nn.Parameter(torch.randn(1, input_dim, hidden_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim * input_dim), nn.Linear(hidden_dim * input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, horizon * len(self.quantiles)))

    def forward(self, x):
        tokens = self.value_proj(x.transpose(1, 2)) + self.var_embed
        h = self.encoder(tokens).reshape(x.shape[0], -1)
        return sorted_quantile_x(self.head(h), self.horizon, self.quantiles)


class TimesBlock(nn.Module):
    def __init__(self, hidden_dim, kernel_size=3, dropout=0.1):
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, kernel_size), padding=(0, pad), groups=hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, periods):
        residual = x
        outs = []
        bsz, seq_len, hidden_dim = x.shape
        for period in periods:
            period = max(1, min(int(period), seq_len))
            pad_len = (period - seq_len % period) % period
            xp = torch.nn.functional.pad(x, (0, 0, 0, pad_len)) if pad_len else x
            rows = xp.shape[1] // period
            xp = xp.reshape(bsz, rows, period, hidden_dim).permute(0, 3, 1, 2)
            yp = self.net(xp).permute(0, 2, 3, 1).reshape(bsz, rows * period, hidden_dim)[:, :seq_len]
            outs.append(yp)
        return self.norm(residual + torch.stack(outs, dim=0).mean(dim=0))


class TimesNetForecaster(nn.Module):
    def __init__(self, input_dim, horizon, hidden_dim=96, dropout=0.1, num_layers=2, top_k=3, quantiles=DEFAULT_QUANTILES):
        super().__init__()
        self.top_k = top_k
        self.horizon = horizon
        self.quantiles = tuple(float(q) for q in quantiles)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([TimesBlock(hidden_dim, dropout=dropout) for _ in range(num_layers)])
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, horizon * len(self.quantiles)))

    def _periods(self, x):
        spec = torch.fft.rfft(x[:, :, 0], dim=1).abs().mean(dim=0)
        if spec.numel() <= 1:
            return [1]
        spec[0] = 0.0
        k = min(self.top_k, spec.numel() - 1)
        freqs = torch.topk(spec, k=k).indices.detach().cpu().numpy()
        seq_len = x.shape[1]
        return [max(1, int(round(seq_len / max(1, f)))) for f in freqs]

    def forward(self, x):
        periods = self._periods(x)
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h, periods)
        return sorted_quantile_x(self.head(h.mean(dim=1)), self.horizon, self.quantiles)


def sinusoidal_embedding(timesteps, dim):
    half = dim // 2
    scale = math.log(10000.0) / max(half - 1, 1)
    frequencies = torch.exp(-scale * torch.arange(half, device=timesteps.device, dtype=torch.float32))
    angles = timesteps.float().unsqueeze(-1) * frequencies.unsqueeze(0)
    embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if dim % 2:
        embedding = torch.nn.functional.pad(embedding, (0, 1))
    return embedding


class ConditionalDiffusionForecaster(nn.Module):
    """Compact TimeDiff/NsDiff-style conditional diffusion forecaster."""

    def __init__(self, input_dim, horizon, hidden_dim=128, diffusion_steps=50, kind="timediff", dropout=0.1):
        super().__init__()
        self.horizon = horizon
        self.diffusion_steps = diffusion_steps
        self.kind = kind
        self.context_encoder = nn.GRU(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=dropout)
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.endpoint_mean = nn.Linear(hidden_dim, horizon)
        self.endpoint_log_scale = nn.Linear(hidden_dim, horizon)
        self.mixup_proj = nn.Linear(horizon, hidden_dim)
        self.time_proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.denoiser = nn.Sequential(
            nn.Linear(horizon + hidden_dim * 3, hidden_dim * 3),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 3, hidden_dim * 3),
            nn.SiLU(),
            nn.Linear(hidden_dim * 3, horizon),
        )
        beta = torch.linspace(1e-4, 0.12, diffusion_steps)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_bar", alpha_bar)

    def context(self, x):
        _, state = self.context_encoder(x)
        return self.context_norm(state[-1])

    def endpoint(self, x, context=None):
        context = self.context(x) if context is None else context
        learned_mean = bounded_x(self.endpoint_mean(context))
        if self.kind == "timediff":
            last = x[:, -1:, 0]
            trend = last - x[:, -2:-1, 0]
            lead = torch.arange(1, self.horizon + 1, device=x.device, dtype=x.dtype).view(1, -1)
            mean = torch.clamp(last + lead * trend, 0.0, 1.2)
        else:
            mean = learned_mean
        scale = torch.nn.functional.softplus(self.endpoint_log_scale(context)) + 0.02
        return mean, scale

    def denoise(self, noisy_residual, step, context, condition):
        t = self.time_proj(sinusoidal_embedding(step, context.shape[-1]))
        mixup = self.mixup_proj(condition)
        return self.denoiser(torch.cat([noisy_residual, context, mixup, t], dim=-1))

    def training_loss(self, x, y):
        context = self.context(x)
        mean, scale = self.endpoint(x, context)
        residual = (y - mean) / scale if self.kind == "nsdiff" else y - mean
        step = torch.randint(0, self.diffusion_steps, (x.shape[0],), device=x.device)
        noise = torch.randn_like(residual)
        alpha_bar = self.alpha_bar[step].unsqueeze(-1)
        noisy = torch.sqrt(alpha_bar) * residual + torch.sqrt(1.0 - alpha_bar) * noise
        if self.kind == "timediff":
            reveal_probability = torch.rand((x.shape[0], 1), device=x.device) * 0.5
            mask = torch.rand_like(y) < reveal_probability
            condition = torch.where(mask, y, mean)
        else:
            condition = mean
        predicted_noise = self.denoise(noisy, step, context, condition)
        diffusion_loss = torch.mean((predicted_noise - noise) ** 2)
        if self.kind == "nsdiff":
            endpoint_loss = (torch.log(scale) + 0.5 * ((y - mean) / scale) ** 2).mean()
        else:
            endpoint_loss = torch.mean(torch.abs(y - mean))
        return diffusion_loss + 0.1 * endpoint_loss

    @torch.no_grad()
    def sample(self, x, n_samples=50):
        context = self.context(x)
        mean, scale = self.endpoint(x, context)
        batch = x.shape[0]
        context = context.repeat_interleave(n_samples, dim=0)
        condition = mean.repeat_interleave(n_samples, dim=0)
        residual = torch.randn(batch * n_samples, self.horizon, device=x.device, dtype=x.dtype)
        for index in reversed(range(self.diffusion_steps)):
            step = torch.full((batch * n_samples,), index, device=x.device, dtype=torch.long)
            predicted_noise = self.denoise(residual, step, context, condition)
            alpha = self.alpha[index]
            alpha_bar = self.alpha_bar[index]
            beta = self.beta[index]
            residual = (residual - (1.0 - alpha) * predicted_noise / torch.sqrt(1.0 - alpha_bar)) / torch.sqrt(alpha)
            if index > 0:
                residual = residual + torch.sqrt(beta) * torch.randn_like(residual)
        if self.kind == "nsdiff":
            values = condition + scale.repeat_interleave(n_samples, dim=0) * residual
        else:
            values = condition + residual
        values = torch.clamp(values, 0.0, 1.2)
        return values.view(batch, n_samples, self.horizon).permute(1, 0, 2)


def _standardize(train_x, *arrays):
    mean = train_x.mean(axis=(0, 1), keepdims=True)
    std = train_x.std(axis=(0, 1), keepdims=True) + 1e-6
    return [(a - mean) / std for a in arrays]


def _device(train_cfg):
    requested = train_cfg["training"].get("device", "auto")
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def prepare_splits(splits):
    mean = splits["train"]["X_hist"].mean(axis=(0, 1), keepdims=True)
    std = splits["train"]["X_hist"].std(axis=(0, 1), keepdims=True) + 1e-6
    train_x, val_x, test_x = [
        (array - mean) / std
        for array in (splits["train"]["X_hist"], splits["validation"]["X_hist"], splits["test"]["X_hist"])
    ]
    return {
        "train": {**splits["train"], "X_hist": train_x.astype("float32"), "_x_mean": mean.astype("float32"), "_x_std": std.astype("float32")},
        "validation": {**splits["validation"], "X_hist": val_x.astype("float32"), "_x_mean": mean.astype("float32"), "_x_std": std.astype("float32")},
        "test": {**splits["test"], "X_hist": test_x.astype("float32"), "_x_mean": mean.astype("float32"), "_x_std": std.astype("float32")},
    }


def _early_stopping_params(train_cfg):
    cfg = train_cfg["training"]
    return int(cfg.get("patience", 10)), float(cfg.get("min_delta", 1e-4))


def _fit_with_early_stopping(model, local, train_cfg, name, loss_fn, batch_size=None):
    device = _device(train_cfg)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["training"]["learning_rate"]), weight_decay=float(train_cfg["training"]["weight_decay"]))
    batch_size = batch_size or int(train_cfg["training"]["batch_size"])
    train_loader = DataLoader(ForecastDataset(local["train"]), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ForecastDataset(local["validation"]), batch_size=batch_size, shuffle=False)
    patience, min_delta = _early_stopping_params(train_cfg)
    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    history = []
    for epoch in range(int(train_cfg["training"]["max_epochs"])):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model, xb, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["training"]["gradient_clip"]))
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_losses.append(float(loss_fn(model, xb, yb).detach().cpu()))
        train_loss = float(np.mean(losses))
        val_loss = float(np.mean(val_losses))
        if val_loss < best_val - min_delta:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
        history.append({"model": name, "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "best_val_loss": best_val, "bad_epochs": bad_epochs})
        print(f"{name} epoch {epoch + 1}/{int(train_cfg['training']['max_epochs'])}: train={train_loss:.6f} val={val_loss:.6f}", flush=True)
        if bad_epochs >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, device


def _deterministic_loss(model, xb, yb):
    pred = model(xb)
    return torch.mean(torch.abs(pred - yb)) + 0.5 * torch.mean((pred - yb) ** 2)


def _quantile_loss_factory(qs):
    def loss_fn(model, xb, yb):
        pred = model(xb)
        err = yb.unsqueeze(-1) - pred
        return torch.maximum(qs * err, (qs - 1.0) * err).mean()
    return loss_fn


def train_deterministic(name, splits, train_cfg, mc_dropout=False):
    device = _device(train_cfg)
    local = prepare_splits(splits)
    torch.manual_seed(int(train_cfg["training"]["seed"]))
    horizon = local["train"]["y_x"].shape[1]
    model = DeterministicSeqForecaster(local["train"]["X_hist"].shape[-1], horizon, mc_dropout=mc_dropout).to(device)
    qs = torch.tensor(model.quantiles, dtype=torch.float32, device=device).view(1, 1, -1)
    model, history, device = _fit_with_early_stopping(model, local, train_cfg, name, _quantile_loss_factory(qs))
    return model, local, history, device


def train_sota_deterministic(name, splits, train_cfg):
    device = _device(train_cfg)
    local = prepare_splits(splits)
    torch.manual_seed(int(train_cfg["training"]["seed"]))
    input_dim = local["train"]["X_hist"].shape[-1]
    history_steps = local["train"]["X_hist"].shape[1]
    horizon = local["train"]["y_x"].shape[1]
    if name == "PatchTST":
        model = PatchTSTForecaster(input_dim, horizon, hidden_dim=64, num_layers=1).to(device)
    elif name == "iTransformer":
        model = ITransformerForecaster(input_dim, history_steps, horizon, hidden_dim=64, num_layers=1).to(device)
    elif name == "TimesNet":
        model = TimesNetForecaster(input_dim, horizon, hidden_dim=64, num_layers=1, top_k=2).to(device)
    else:
        raise ValueError(f"Unknown SOTA forecaster: {name}")
    batch_size = max(int(train_cfg["training"]["batch_size"]), 256)
    qs = torch.tensor(model.quantiles, dtype=torch.float32, device=device).view(1, 1, -1)
    model, history, device = _fit_with_early_stopping(model, local, train_cfg, name, _quantile_loss_factory(qs), batch_size=batch_size)
    return model, local, history, device


def train_quantile(splits, train_cfg):
    device = _device(train_cfg)
    local = prepare_splits(splits)
    torch.manual_seed(int(train_cfg["training"]["seed"]))
    horizon = local["train"]["y_x"].shape[1]
    model = QuantileSeqForecaster(local["train"]["X_hist"].shape[-1], horizon).to(device)
    qs = torch.tensor(model.quantiles, dtype=torch.float32, device=device).view(1, 1, -1)
    model, history, device = _fit_with_early_stopping(model, local, train_cfg, "QuantileGRU", _quantile_loss_factory(qs))
    return model, local, history, device


def train_diffusion(name, splits, train_cfg, diffusion_steps=50):
    device = _device(train_cfg)
    local = prepare_splits(splits)
    torch.manual_seed(int(train_cfg["training"]["seed"]))
    kind = "nsdiff" if name == "NsDiff-style" else "timediff"
    model = ConditionalDiffusionForecaster(
        local["train"]["X_hist"].shape[-1],
        local["train"]["y_x"].shape[1],
        hidden_dim=128,
        diffusion_steps=diffusion_steps,
        kind=kind,
    ).to(device)

    def loss_fn(current_model, xb, yb):
        return current_model.training_loss(xb, yb)

    model, history, device = _fit_with_early_stopping(model, local, train_cfg, name, loss_fn)
    return model, local, history, device


def predict_deterministic(model, split, device, mc_samples=0):
    x = torch.tensor(split["X_hist"], dtype=torch.float32, device=device)
    if mc_samples > 0:
        model.train()
        with torch.no_grad():
            q_samples = torch.stack([model(x) for _ in range(mc_samples)], dim=0).cpu().numpy()
        median_idx = list(model.quantiles).index(0.5)
        median_samples = q_samples[:, :, :, median_idx]
        return {
            "mean_x": median_samples.mean(axis=0),
            "samples_x": median_samples,
            "quantiles": model.quantiles,
            "quantile_x": q_samples.mean(axis=0),
        }
    model.eval()
    with torch.no_grad():
        pred = model(x)
    if hasattr(model, "quantiles"):
        q = pred.cpu().numpy()
        median_idx = list(model.quantiles).index(0.5)
        return {"mean_x": q[:, :, median_idx], "quantiles": model.quantiles, "quantile_x": q}
    return {"mean_x": pred.cpu().numpy()}


def predict_quantile(model, split, device):
    model.eval()
    x = torch.tensor(split["X_hist"], dtype=torch.float32, device=device)
    with torch.no_grad():
        q = model(x).cpu().numpy()
    median_idx = list(model.quantiles).index(0.5)
    return {"mean_x": q[:, :, median_idx], "quantiles": model.quantiles, "quantile_x": q}


def predict_diffusion(model, split, device, n_samples=50, batch_size=256):
    model.eval()
    samples = []
    x = torch.tensor(split["X_hist"], dtype=torch.float32)
    for start in range(0, len(x), batch_size):
        xb = x[start:start + batch_size].to(device)
        samples.append(model.sample(xb, n_samples=n_samples).cpu())
    sample_x = torch.cat(samples, dim=1).numpy()
    return {"mean_x": sample_x.mean(axis=0), "samples_x": sample_x}


def fit_ramp_residual_std(mean_x, split):
    power_mean = mean_x * split["p_cs_next"]
    capacity = split["capacity_kw"]
    # For multi-step ramp prediction, only the first "previous" power is observed.
    # For later horizons, use the model's prior-step prediction as the previous power.
    if split["p_prev"].ndim == 2:
        prev0 = split["p_prev"][:, 0]
    else:
        prev0 = split["p_prev"]
    prev_pred = np.zeros_like(power_mean, dtype="float32")
    prev_pred[:, 0] = prev0.astype("float32")
    if power_mean.shape[1] > 1:
        prev_pred[:, 1:] = power_mean[:, :-1]
    mean_ramp = (power_mean - prev_pred) / capacity
    actual_ramp = (split["y_power"] - split["p_prev"]) / capacity
    return np.std(actual_ramp - mean_ramp, axis=0) + 1e-4


def event_probs_from_gaussian_power(mean_x, std_x, split, thresholds):
    power_mean = mean_x * split["p_cs_next"]
    power_std = np.maximum(std_x * split["p_cs_next"], 1e-5)
    capacity = split["capacity_kw"]
    if split["p_prev"].ndim == 2:
        prev0 = split["p_prev"][:, 0]
    else:
        prev0 = split["p_prev"]
    prev_mean = np.zeros_like(power_mean, dtype="float32")
    prev_mean[:, 0] = prev0.astype("float32")
    if power_mean.shape[1] > 1:
        prev_mean[:, 1:] = power_mean[:, :-1]
    prev_std = np.zeros_like(power_std, dtype="float32")
    if power_std.shape[1] > 1:
        prev_std[:, 1:] = power_std[:, :-1]
    ramp_mean = (power_mean - prev_mean) / capacity
    # Independence approximation for consecutive-step uncertainty in delta power.
    ramp_std = np.sqrt(power_std ** 2 + prev_std ** 2) / capacity
    down = np.stack([ndtr((-float(g) - ramp_mean) / ramp_std) for g in thresholds], axis=-1)
    up = np.stack([1.0 - ndtr((float(g) - ramp_mean) / ramp_std) for g in thresholds], axis=-1)
    return down, up


def event_probs_from_residual(mean_x, split, thresholds, residual_std):
    # residual_std is in ramp-fraction units per horizon (already capacity-normalized).
    power_mean = mean_x * split["p_cs_next"]
    capacity = split["capacity_kw"]
    if split["p_prev"].ndim == 2:
        prev0 = split["p_prev"][:, 0]
    else:
        prev0 = split["p_prev"]
    prev_pred = np.zeros_like(power_mean, dtype="float32")
    prev_pred[:, 0] = prev0.astype("float32")
    if power_mean.shape[1] > 1:
        prev_pred[:, 1:] = power_mean[:, :-1]
    ramp_mean = (power_mean - prev_pred) / capacity
    ramp_std = np.maximum(np.asarray(residual_std, dtype="float32").reshape(1, -1), 1e-5)
    down = np.stack([ndtr((-float(g) - ramp_mean) / ramp_std) for g in thresholds], axis=-1)
    up = np.stack([1.0 - ndtr((float(g) - ramp_mean) / ramp_std) for g in thresholds], axis=-1)
    return down, up


def gaussian_crps(y, mean, std):
    std = np.maximum(std, 1e-6)
    z = (y - mean) / std
    pdf = np.exp(-0.5 * z ** 2) / math.sqrt(2.0 * math.pi)
    cdf = ndtr(z)
    return std * (z * (2.0 * cdf - 1.0) + 2.0 * pdf - 1.0 / math.sqrt(math.pi))


def sample_crps(y, samples):
    # samples: S, N, H
    term1 = np.mean(np.abs(samples - y[None, :, :]), axis=0)
    sorted_samples = np.sort(samples, axis=0)
    s = samples.shape[0]
    weights = (2 * np.arange(1, s + 1).reshape(s, 1, 1) - s - 1)
    term2 = np.sum(weights * sorted_samples, axis=0) / (s * s)
    return term1 - term2
