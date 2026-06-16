import math
import torch
from torch import nn
import torch.nn.functional as F


DEFAULT_QUANTILES = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)


def bounded_quantiles(raw, forecast_steps, quantiles, x_max):
    q = x_max * torch.sigmoid(raw.view(-1, forecast_steps, len(quantiles)))
    return torch.sort(q, dim=-1).values


def bounded_x(raw, x_max=1.2):
    return float(x_max) * torch.sigmoid(raw)


def anchored_quantiles(raw, anchor, forecast_steps, quantiles, x_min, x_max, residual_scale):
    residual = float(residual_scale) * torch.tanh(raw.view(-1, forecast_steps, len(quantiles)))
    q = torch.clamp(anchor.unsqueeze(-1) + residual, float(x_min), float(x_max))
    return torch.sort(q, dim=-1).values


def smart_persistence_anchor(x_hist, forecast_steps, x_min, x_max):
    last = x_hist[:, -1, 0:1]
    if x_hist.shape[1] > 1:
        trend = last - x_hist[:, -2:-1, 0]
    else:
        trend = torch.zeros_like(last)
    horizon = torch.arange(1, forecast_steps + 1, device=x_hist.device, dtype=x_hist.dtype).view(1, -1)
    return torch.clamp(last + horizon * trend, float(x_min), float(x_max))


def persistence_anchor(x_hist, forecast_steps, x_min, x_max):
    last = x_hist[:, -1, 0:1]
    return torch.clamp(last.expand(-1, forecast_steps), float(x_min), float(x_max))


class JumpRS(nn.Module):
    """Main JumpRS model: signed regime jump-diffusion conditioned on clear-sky geometry."""

    uses_future_context = True

    def __init__(
        self,
        input_dim,
        history_steps,
        forecast_steps=1,
        hidden_dim=128,
        num_layers=2,
        dropout=0.1,
        num_regimes=3,
        x_min=0.0,
        x_max=1.2,
        m_max=0.5,
        eps=1e-6,
        nhead=4,
        quantiles=DEFAULT_QUANTILES,
        quantile_residual_scale=0.35,
        point_residual_scale=0.35,
        use_forecast_heads=False,
        forecast_head_anchor="persistence",
        forecast_head_arch="attention",
        dedicated_predictor=False,
        site_conditioning=False,
        markov_regimes=False,
        belief_filtering=False,
        state_conditioned_decoder=False,
        **_,
    ):
        super().__init__()
        self.x_min = x_min
        self.x_max = x_max
        self.m_max = m_max
        self.eps = eps
        self.num_regimes = num_regimes
        self.forecast_steps = forecast_steps
        self.quantiles = tuple(float(q) for q in quantiles)
        self.quantile_residual_scale = quantile_residual_scale
        self.point_residual_scale = point_residual_scale
        self.use_forecast_heads = bool(use_forecast_heads)
        self.forecast_head_anchor = str(forecast_head_anchor)
        self.forecast_head_arch = str(forecast_head_arch)
        self.dedicated_predictor = bool(dedicated_predictor)
        self.site_conditioning = bool(site_conditioning)
        self.markov_regimes = bool(markov_regimes)
        self.belief_filtering = bool(belief_filtering)
        self.state_conditioned_decoder = bool(state_conditioned_decoder)
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.multiscale_convs = nn.ModuleList(
            [
                nn.Conv1d(input_dim, hidden_dim // 4, kernel_size=kernel, padding=kernel - 1)
                for kernel in (2, 4, 8)
            ]
        )
        self.multiscale_proj = nn.Linear(3 * (hidden_dim // 4), hidden_dim)
        self.history_stats_proj = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.site_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.pos_embed = nn.Parameter(torch.randn(1, history_steps, hidden_dim) * 0.02)
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
        self.encoder_norm = nn.LayerNorm(hidden_dim)
        self.horizon_queries = nn.Parameter(torch.randn(1, forecast_steps, hidden_dim) * 0.02)
        self.horizon_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout, batch_first=True)
        self.horizon_ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.horizon_norm = nn.LayerNorm(hidden_dim)
        self.solar_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.param_head = nn.Linear(hidden_dim, num_regimes * 9)
        self.regime_head = nn.Linear(hidden_dim, num_regimes)
        self.regime_transition_head = nn.Linear(hidden_dim, num_regimes * num_regimes)
        self.initial_belief_logits = nn.Parameter(torch.zeros(num_regimes))
        self.history_transition_logits = nn.Parameter(torch.eye(num_regimes) * 2.0)
        self.diffusion_center = nn.Parameter(torch.linspace(-0.03, 0.03, num_regimes))
        self.diffusion_log_scale = nn.Parameter(torch.full((num_regimes,), -2.5))
        self.jump_logits = nn.Parameter(torch.zeros(num_regimes, 3))
        self.jump_center = nn.Parameter(
            torch.stack(
                [
                    torch.linspace(-0.12, -0.04, num_regimes),
                    torch.zeros(num_regimes),
                    torch.linspace(0.04, 0.12, num_regimes),
                ],
                dim=-1,
            )
        )
        self.jump_log_scale = nn.Parameter(torch.full((num_regimes, 3), -2.2))
        self.belief_embed = nn.Sequential(
            nn.Linear(num_regimes, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.state_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.register_buffer("loss_ema_nll", torch.tensor(1.0))
        self.register_buffer("loss_ema_crps", torch.tensor(1.0))
        self.point_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.quantile_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(self.quantiles)),
        )
        self.forecast_gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        self.forecast_gru_norm = nn.LayerNorm(hidden_dim)
        self.direct_point_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, forecast_steps),
        )
        self.direct_quantile_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, forecast_steps * len(self.quantiles)),
        )

    def _dedicated_history_features(self, x_hist):
        conv_input = x_hist.transpose(1, 2)
        branches = [
            conv(conv_input)[..., : x_hist.shape[1]].transpose(1, 2)
            for conv in self.multiscale_convs
        ]
        x_state = x_hist[..., 0]
        dx = x_state[:, 1:] - x_state[:, :-1]
        stats = torch.stack(
            [
                x_state.mean(dim=1),
                x_state.std(dim=1),
                x_state[:, -1],
                dx[:, -1],
                dx.abs().mean(dim=1),
                dx.std(dim=1),
            ],
            dim=-1,
        )
        return self.multiscale_proj(torch.cat(branches, dim=-1)), self.history_stats_proj(stats)

    def _markov_regime_log_probs(self, h):
        emissions = torch.softmax(self.regime_head(h), dim=-1)
        transitions = torch.softmax(
            self.regime_transition_head(h).view(
                h.shape[0], self.forecast_steps, self.num_regimes, self.num_regimes
            ),
            dim=-1,
        )
        probs = [emissions[:, 0]]
        for tau in range(1, self.forecast_steps):
            prior = torch.bmm(probs[-1].unsqueeze(1), transitions[:, tau]).squeeze(1)
            posterior = prior * emissions[:, tau]
            probs.append(posterior / torch.clamp(posterior.sum(dim=-1, keepdim=True), min=self.eps))
        return torch.log(torch.clamp(torch.stack(probs, dim=1), min=self.eps))

    def _filter_history_belief(self, x_hist):
        """Low-dimensional Zakai-style filter using diffusion and signed-jump innovations."""
        batch = x_hist.shape[0]
        belief = torch.softmax(self.initial_belief_logits, dim=-1).expand(batch, -1)
        transition = torch.softmax(self.history_transition_logits, dim=-1)
        dx = x_hist[:, 1:, 0] - x_hist[:, :-1, 0]
        diffusion_scale = F.softplus(self.diffusion_log_scale) + self.eps
        jump_scale = F.softplus(self.jump_log_scale) + self.eps
        jump_log_weight = torch.log_softmax(self.jump_logits, dim=-1)
        filter_nll = torch.zeros(batch, device=x_hist.device, dtype=x_hist.dtype)
        belief_kl = torch.zeros_like(filter_nll)
        for tau in range(dx.shape[1]):
            prior = belief @ transition
            innovation = dx[:, tau].unsqueeze(-1)
            diffusion_log_like = (
                -torch.log(diffusion_scale)
                - 0.5 * ((innovation - self.diffusion_center) / diffusion_scale).pow(2)
            )
            jump_innovation = innovation.unsqueeze(-1)
            jump_log_like = torch.logsumexp(
                jump_log_weight
                - torch.log(jump_scale)
                - 0.5 * ((jump_innovation - self.jump_center) / jump_scale).pow(2),
                dim=-1,
            )
            log_belief = (
                torch.log(torch.clamp(prior, min=self.eps))
                + diffusion_log_like
                + jump_log_like
            )
            log_evidence = torch.logsumexp(log_belief, dim=-1)
            belief = torch.softmax(log_belief, dim=-1)
            filter_nll = filter_nll - log_evidence
            belief_kl = belief_kl + (
                belief
                * (
                    torch.log(torch.clamp(belief, min=self.eps))
                    - torch.log(torch.clamp(prior, min=self.eps))
                )
            ).sum(dim=-1)
        steps = max(dx.shape[1], 1)
        return belief, filter_nll / steps, belief_kl / steps

    def _decode_belief_step(self, h, z, belief):
        """Propagate belief without future observations and decode a state-conditioned kernel."""
        transitions = torch.softmax(
            self.regime_transition_head(h).view(-1, self.num_regimes, self.num_regimes),
            dim=-1,
        )
        next_belief = torch.bmm(belief.unsqueeze(1), transitions).squeeze(1)
        next_belief = next_belief / torch.clamp(next_belief.sum(dim=-1, keepdim=True), min=self.eps)
        decoded_h = h + self.belief_embed(next_belief)
        if self.state_conditioned_decoder:
            decoded_h = decoded_h + self.state_embed(z.unsqueeze(-1))
        params = decode_signed_regime_params(
            self.param_head(decoded_h),
            torch.log(torch.clamp(next_belief, min=self.eps)),
            1,
            self.num_regimes,
            self.x_min,
            self.x_max,
            self.m_max,
            self.eps,
        )
        return {key: value[:, 0] for key, value in params.items()}, next_belief

    def _belief_conditioned_params(self, h, x_hist):
        belief, filter_nll, belief_kl = self._filter_history_belief(x_hist)
        initial_belief = belief
        z = x_hist[:, -1, 0]
        steps = []
        for tau in range(self.forecast_steps):
            step, belief = self._decode_belief_step(h[:, tau], z, belief)
            steps.append(step)
            z = torch.clamp(
                z + step["kappa"] * (step["xbar"] - z),
                self.x_min,
                self.x_max,
            )
        params = {
            key: torch.stack([step[key] for step in steps], dim=1)
            for key in steps[0]
        }
        params["_rollout_h"] = h
        params["_rollout_belief0"] = initial_belief
        params["_filter_nll"] = filter_nll
        params["_belief_kl"] = belief_kl
        return params

    def forward(self, x_hist, future_clear_sky=None, site_context=None):
        tokens = self.input_proj(self.input_norm(x_hist)) + self.pos_embed[:, : x_hist.shape[1]]
        history_summary = None
        if self.dedicated_predictor:
            multiscale, history_summary = self._dedicated_history_features(x_hist)
            tokens = tokens + multiscale
        tokens = self.encoder_norm(self.encoder(tokens))
        queries = self.horizon_queries.expand(x_hist.shape[0], -1, -1)
        if future_clear_sky is not None:
            queries = queries + self.solar_embed(future_clear_sky.unsqueeze(-1))
        if history_summary is not None:
            queries = queries + history_summary.unsqueeze(1)
        if self.site_conditioning and site_context is not None:
            queries = queries + self.site_embed(site_context.reshape(-1, 1)).unsqueeze(1)
        decoded, _ = self.horizon_attn(queries, tokens, tokens, need_weights=False)
        h = queries + decoded
        h = self.horizon_norm(h + self.horizon_ffn(h))
        if self.belief_filtering:
            params = self._belief_conditioned_params(h, x_hist)
        else:
            regime_logits = self._markov_regime_log_probs(h) if self.markov_regimes else self.regime_head(h)
            params = decode_signed_regime_params(
                self.param_head(h).reshape(x_hist.shape[0], -1),
                regime_logits.reshape(x_hist.shape[0], -1),
                self.forecast_steps,
                self.num_regimes,
                self.x_min,
                self.x_max,
                self.m_max,
                self.eps,
            )
        if self.use_forecast_heads:
            if self.forecast_head_arch in {"gru", "gru_direct"}:
                _, h_gru = self.forecast_gru(x_hist)
                h_last = self.forecast_gru_norm(h_gru[-1])
                if self.forecast_head_arch == "gru_direct":
                    point = bounded_x(self.direct_point_head(h_last), self.x_max)
                    q = bounded_quantiles(
                        self.direct_quantile_head(h_last),
                        self.forecast_steps,
                        self.quantiles,
                        self.x_max,
                    )
                    params["point_mean_x"] = point
                    params["point_quantile_x"] = q
                    return params
                h_forecast = h_last.unsqueeze(1).expand(-1, self.forecast_steps, -1)
                if future_clear_sky is not None:
                    h_forecast = h_forecast + self.solar_embed(future_clear_sky.unsqueeze(-1))
            else:
                h_forecast = h
            if self.forecast_head_anchor == "bounded":
                point = bounded_x(self.point_head(h_forecast).squeeze(-1), self.x_max)
                q = bounded_quantiles(
                    self.quantile_head(h_forecast).reshape(x_hist.shape[0], -1),
                    self.forecast_steps,
                    self.quantiles,
                    self.x_max,
                )
            else:
                anchor = persistence_anchor(x_hist, self.forecast_steps, self.x_min, self.x_max)
                point_delta = self.point_residual_scale * torch.tanh(self.point_head(h_forecast).squeeze(-1))
                point = torch.clamp(anchor + point_delta, self.x_min, self.x_max)
                q = anchored_quantiles(
                    self.quantile_head(h_forecast).reshape(x_hist.shape[0], -1),
                    point,
                    self.forecast_steps,
                    self.quantiles,
                    self.x_min,
                    self.x_max,
                    self.quantile_residual_scale,
                )
            params["point_mean_x"] = point
            params["point_quantile_x"] = q
        return params


def decode_params(raw_params, raw_regime, forecast_steps, num_regimes, x_min, x_max, m_max, eps):
    pi = torch.softmax(raw_regime.view(-1, forecast_steps, num_regimes), dim=-1)
    raw = raw_params.view(-1, forecast_steps, num_regimes, 6)
    kappa = F.softplus(raw[..., 0]) + eps
    xbar = x_min + (x_max - x_min) * torch.sigmoid(raw[..., 1])
    sigma = F.softplus(raw[..., 2]) + eps
    lam = F.softplus(raw[..., 3])
    mu_j = m_max * torch.tanh(raw[..., 4])
    eta_j = F.softplus(raw[..., 5]) + eps
    mix = lambda a: (pi * a).sum(dim=-1)
    return {
        "kappa": mix(kappa),
        "xbar": mix(xbar),
        "sigma": mix(sigma),
        "lambda": mix(lam),
        "mu_j": mix(mu_j),
        "eta_j": mix(eta_j),
        "pi": pi,
    }


def decode_regime_params(raw_params, raw_regime, forecast_steps, num_regimes, x_min, x_max, m_max, eps):
    """Decode regime kernels without collapsing them before nonlinear mixing."""
    pi = torch.softmax(raw_regime.view(-1, forecast_steps, num_regimes), dim=-1)
    raw = raw_params.view(-1, forecast_steps, num_regimes, 6)
    regime = {
        "kappa": F.softplus(raw[..., 0]) + eps,
        "xbar": x_min + (x_max - x_min) * torch.sigmoid(raw[..., 1]),
        "sigma": F.softplus(raw[..., 2]) + eps,
        "lambda": F.softplus(raw[..., 3]),
        "mu_j": m_max * torch.tanh(raw[..., 4]),
        "eta_j": F.softplus(raw[..., 5]) + eps,
    }
    out = {"pi": pi}
    for key, value in regime.items():
        out[key] = (pi * value).sum(dim=-1)
        out[f"regime_{key}"] = value
    return out


def decode_signed_params(raw_params, raw_regime, forecast_steps, num_regimes, x_min, x_max, m_max, eps):
    pi = torch.softmax(raw_regime.view(-1, forecast_steps, num_regimes), dim=-1)
    raw = raw_params.view(-1, forecast_steps, num_regimes, 9)
    kappa = F.softplus(raw[..., 0]) + eps
    xbar = x_min + (x_max - x_min) * torch.sigmoid(raw[..., 1])
    sigma = F.softplus(raw[..., 2]) + eps
    lam_down = F.softplus(raw[..., 3])
    lam_up = F.softplus(raw[..., 4])
    mu_down = -m_max * torch.sigmoid(raw[..., 5])
    mu_up = m_max * torch.sigmoid(raw[..., 6])
    eta_down = F.softplus(raw[..., 7]) + eps
    eta_up = F.softplus(raw[..., 8]) + eps
    mix = lambda a: (pi * a).sum(dim=-1)
    lam_down_m = mix(lam_down)
    lam_up_m = mix(lam_up)
    total_lam = lam_down_m + lam_up_m
    return {
        "kappa": mix(kappa),
        "xbar": mix(xbar),
        "sigma": mix(sigma),
        "lambda_down": lam_down_m,
        "lambda_up": lam_up_m,
        "mu_down": mix(mu_down),
        "mu_up": mix(mu_up),
        "eta_down": mix(eta_down),
        "eta_up": mix(eta_up),
        "lambda": total_lam,
        "mu_j": (lam_down_m * mix(mu_down) + lam_up_m * mix(mu_up)) / torch.clamp(total_lam, min=eps),
        "eta_j": 0.5 * (mix(eta_down) + mix(eta_up)),
        "pi": pi,
    }


def decode_signed_regime_params(raw_params, raw_regime, forecast_steps, num_regimes, x_min, x_max, m_max, eps):
    pi = torch.softmax(raw_regime.view(-1, forecast_steps, num_regimes), dim=-1)
    raw = raw_params.view(-1, forecast_steps, num_regimes, 9)
    regime = {
        "kappa": F.softplus(raw[..., 0]) + eps,
        "xbar": x_min + (x_max - x_min) * torch.sigmoid(raw[..., 1]),
        "sigma": F.softplus(raw[..., 2]) + eps,
        "lambda_down": F.softplus(raw[..., 3]),
        "lambda_up": F.softplus(raw[..., 4]),
        "mu_down": -m_max * torch.sigmoid(raw[..., 5]),
        "mu_up": m_max * torch.sigmoid(raw[..., 6]),
        "eta_down": F.softplus(raw[..., 7]) + eps,
        "eta_up": F.softplus(raw[..., 8]) + eps,
    }
    out = {"pi": pi}
    for key, value in regime.items():
        out[key] = (pi * value).sum(dim=-1)
        out[f"regime_{key}"] = value
    out["lambda"] = out["lambda_down"] + out["lambda_up"]
    out["mu_j"] = (
        out["lambda_down"] * out["mu_down"] + out["lambda_up"] * out["mu_up"]
    ) / torch.clamp(out["lambda"], min=eps)
    out["eta_j"] = 0.5 * (out["eta_down"] + out["eta_up"])
    return out


def jacobi_loading(z, x_min, x_max):
    return torch.sqrt(torch.clamp((z - x_min) * (x_max - z), min=0.0))


def _poisson_log_weights(lam_dt, counts):
    log_w = -lam_dt + counts * torch.log(torch.clamp(lam_dt, min=1e-12)) - torch.lgamma(counts + 1.0)
    return log_w - torch.logsumexp(log_w, dim=-1, keepdim=True)


def transition_components(params, z, model_cfg):
    cfg = model_cfg["model"]
    nu = int(cfg["truncation_order"])
    dt = float(cfg["dt"])
    eps_var = float(cfg["eps_var"])
    if cfg.get("diffusion_loading", "jacobi") == "constant":
        g = torch.ones_like(z).unsqueeze(-1)
    else:
        g = jacobi_loading(z, float(cfg["x_min"]), float(cfg["x_max"])).unsqueeze(-1)
    if "regime_lambda_down" in params:
        counts = torch.arange(nu + 1, device=z.device, dtype=z.dtype)
        n_down, n_up = torch.meshgrid(counts, counts, indexing="ij")
        n_down = n_down.reshape(-1).view(*([1] * z.ndim), 1, -1)
        n_up = n_up.reshape(-1).view(*([1] * z.ndim), 1, -1)
        z_r = z.unsqueeze(-1).unsqueeze(-1)
        g_r = g.unsqueeze(-1)
        pi = params["pi"]
        kappa = params["regime_kappa"]
        xbar = params["regime_xbar"]
        sigma = params["regime_sigma"]
        lam_down = params["regime_lambda_down"]
        lam_up = params["regime_lambda_up"]
        mu_down = params["regime_mu_down"]
        mu_up = params["regime_mu_up"]
        eta_down = params["regime_eta_down"]
        eta_up = params["regime_eta_up"]
        if bool(cfg.get("disable_jump", False)):
            lam_down = torch.zeros_like(lam_down)
            lam_up = torch.zeros_like(lam_up)
            mu_down = torch.zeros_like(mu_down)
            mu_up = torch.zeros_like(mu_up)
            eta_down = torch.zeros_like(eta_down)
            eta_up = torch.zeros_like(eta_up)
        lam_down_dt = lam_down.unsqueeze(-1) * dt
        lam_up_dt = lam_up.unsqueeze(-1) * dt
        log_w = (
            torch.log(torch.clamp(pi, min=1e-12)).unsqueeze(-1)
            - lam_down_dt
            + n_down * torch.log(torch.clamp(lam_down_dt, min=1e-12))
            - torch.lgamma(n_down + 1.0)
            - lam_up_dt
            + n_up * torch.log(torch.clamp(lam_up_dt, min=1e-12))
            - torch.lgamma(n_up + 1.0)
        )
        mean = (
            z_r
            + (
                kappa.unsqueeze(-1) * (xbar.unsqueeze(-1) - z_r)
                - lam_down.unsqueeze(-1) * mu_down.unsqueeze(-1)
                - lam_up.unsqueeze(-1) * mu_up.unsqueeze(-1)
            )
            * dt
            + n_down * mu_down.unsqueeze(-1)
            + n_up * mu_up.unsqueeze(-1)
        )
        var = (
            sigma.unsqueeze(-1).pow(2) * g_r.pow(2) * dt
            + n_down * eta_down.unsqueeze(-1).pow(2)
            + n_up * eta_up.unsqueeze(-1).pow(2)
            + eps_var ** 2
        )
        component_count = pi.shape[-1] * (nu + 1) ** 2
        log_w = log_w.reshape(*z.shape, component_count)
        log_w = log_w - torch.logsumexp(log_w, dim=-1, keepdim=True)
        mean = mean.reshape(*z.shape, component_count)
        std = torch.sqrt(torch.clamp(var, min=eps_var ** 2)).reshape(*z.shape, component_count)
        return log_w, mean, std
    if "regime_kappa" in params:
        counts = torch.arange(nu + 1, device=z.device, dtype=z.dtype)
        n = counts.view(*([1] * z.ndim), 1, -1)
        z_r = z.unsqueeze(-1).unsqueeze(-1)
        g_r = g.unsqueeze(-1)
        pi = params["pi"]
        kappa = params["regime_kappa"]
        xbar = params["regime_xbar"]
        sigma = params["regime_sigma"]
        lam = params["regime_lambda"]
        mu_j = params["regime_mu_j"]
        eta_j = params["regime_eta_j"]
        if bool(cfg.get("disable_jump", False)):
            lam = torch.zeros_like(lam)
            mu_j = torch.zeros_like(mu_j)
            eta_j = torch.zeros_like(eta_j)
        lam_dt = lam.unsqueeze(-1) * dt
        log_w = (
            torch.log(torch.clamp(pi, min=1e-12)).unsqueeze(-1)
            - lam_dt
            + n * torch.log(torch.clamp(lam_dt, min=1e-12))
            - torch.lgamma(n + 1.0)
        )
        mean = (
            z_r
            + (
                kappa.unsqueeze(-1) * (xbar.unsqueeze(-1) - z_r)
                - lam.unsqueeze(-1) * mu_j.unsqueeze(-1)
            )
            * dt
            + n * mu_j.unsqueeze(-1)
        )
        var = (
            sigma.unsqueeze(-1).pow(2) * g_r.pow(2) * dt
            + n * eta_j.unsqueeze(-1).pow(2)
            + eps_var ** 2
        )
        component_count = pi.shape[-1] * (nu + 1)
        log_w = log_w.reshape(*z.shape, component_count)
        log_w = log_w - torch.logsumexp(log_w, dim=-1, keepdim=True)
        mean = mean.reshape(*z.shape, component_count)
        std = torch.sqrt(torch.clamp(var, min=eps_var ** 2)).reshape(*z.shape, component_count)
        return log_w, mean, std
    if "lambda_down" in params and "lambda_up" in params:
        counts = torch.arange(nu + 1, device=z.device, dtype=z.dtype)
        n_down, n_up = torch.meshgrid(counts, counts, indexing="ij")
        n_down = n_down.reshape(-1).view(*([1] * z.ndim), -1)
        n_up = n_up.reshape(-1).view(*([1] * z.ndim), -1)
        lam_down = params["lambda_down"]
        lam_up = params["lambda_up"]
        mu_down = params["mu_down"]
        mu_up = params["mu_up"]
        eta_down = params["eta_down"]
        eta_up = params["eta_up"]
        if bool(cfg.get("disable_jump", False)):
            lam_down = torch.zeros_like(lam_down)
            lam_up = torch.zeros_like(lam_up)
            mu_down = torch.zeros_like(mu_down)
            mu_up = torch.zeros_like(mu_up)
            eta_down = torch.zeros_like(eta_down)
            eta_up = torch.zeros_like(eta_up)
        lam_down_dt = lam_down.unsqueeze(-1) * dt
        lam_up_dt = lam_up.unsqueeze(-1) * dt
        log_w = (
            -lam_down_dt
            + n_down * torch.log(torch.clamp(lam_down_dt, min=1e-12))
            - torch.lgamma(n_down + 1.0)
            - lam_up_dt
            + n_up * torch.log(torch.clamp(lam_up_dt, min=1e-12))
            - torch.lgamma(n_up + 1.0)
        )
        log_w = log_w - torch.logsumexp(log_w, dim=-1, keepdim=True)
        mean = (
            z.unsqueeze(-1)
            + (
                params["kappa"].unsqueeze(-1) * (params["xbar"].unsqueeze(-1) - z.unsqueeze(-1))
                - lam_down.unsqueeze(-1) * mu_down.unsqueeze(-1)
                - lam_up.unsqueeze(-1) * mu_up.unsqueeze(-1)
            )
            * dt
            + n_down * mu_down.unsqueeze(-1)
            + n_up * mu_up.unsqueeze(-1)
        )
        var = (
            (params["sigma"].unsqueeze(-1) ** 2) * (g ** 2) * dt
            + n_down * (eta_down.unsqueeze(-1) ** 2)
            + n_up * (eta_up.unsqueeze(-1) ** 2)
            + eps_var ** 2
        )
        return log_w, mean, torch.sqrt(torch.clamp(var, min=eps_var ** 2))
    n = torch.arange(nu + 1, device=z.device, dtype=z.dtype).view(*([1] * z.ndim), -1)
    lam = params["lambda"]
    mu_j = params["mu_j"]
    eta_j = params["eta_j"]
    if bool(cfg.get("disable_jump", False)):
        lam = torch.zeros_like(lam)
        mu_j = torch.zeros_like(mu_j)
        eta_j = torch.zeros_like(eta_j)
    log_w = _poisson_log_weights(lam.unsqueeze(-1) * dt, n)
    mean = z.unsqueeze(-1) + (params["kappa"].unsqueeze(-1) * (params["xbar"].unsqueeze(-1) - z.unsqueeze(-1)) - lam.unsqueeze(-1) * mu_j.unsqueeze(-1)) * dt + n * mu_j.unsqueeze(-1)
    var = (params["sigma"].unsqueeze(-1) ** 2) * (g ** 2) * dt + n * (eta_j.unsqueeze(-1) ** 2) + eps_var ** 2
    return log_w, mean, torch.sqrt(torch.clamp(var, min=eps_var ** 2))


def transition_log_prob(x, z, params, model_cfg):
    log_w, mean, std = transition_components(params, z, model_cfg)
    x = x.unsqueeze(-1)
    log_norm = -0.5 * math.log(2 * math.pi) - torch.log(std) - 0.5 * ((x - mean) / std) ** 2
    return torch.logsumexp(log_w + log_norm, dim=-1)


def mean_center(z, params, model_cfg):
    cfg = model_cfg["model"]
    m = z + params["kappa"] * (params["xbar"] - z) * float(cfg["dt"])
    return torch.clamp(m, float(cfg["x_min"]), float(cfg["x_max"]))


def one_step_ramp_probabilities(z, p_cs_prev, p_cs_next, capacity_kw, thresholds, params, model_cfg):
    if z.ndim == 2:
        if p_cs_prev.ndim == 1:
            p_cs_prev = p_cs_prev.unsqueeze(1)
        if p_cs_next.ndim == 1:
            p_cs_next = p_cs_next.unsqueeze(1)
        if torch.is_tensor(capacity_kw) and capacity_kw.ndim == 1:
            capacity_kw = capacity_kw.unsqueeze(1)
    log_w, mean, std = transition_components(params, z, model_cfg)
    weights = torch.exp(log_w)
    down_probs, up_probs = [], []
    normal = torch.distributions.Normal(torch.tensor(0.0, device=z.device), torch.tensor(1.0, device=z.device))
    for gamma in thresholds:
        c_down = (p_cs_prev * z - float(gamma) * capacity_kw) / torch.clamp(p_cs_next, min=1e-6)
        c_up = (p_cs_prev * z + float(gamma) * capacity_kw) / torch.clamp(p_cs_next, min=1e-6)
        down = (weights * normal.cdf((c_down.unsqueeze(-1) - mean) / std)).sum(dim=-1)
        up = (weights * (1.0 - normal.cdf((c_up.unsqueeze(-1) - mean) / std))).sum(dim=-1)
        down_probs.append(down)
        up_probs.append(up)
    return torch.stack(down_probs, dim=-1), torch.stack(up_probs, dim=-1)


def rollout_mean_centers(z0, params, model_cfg):
    centers = []
    z = z0
    horizon = params["kappa"].shape[1]
    for tau in range(horizon):
        step_params = {
            key: value[:, tau]
            for key, value in params.items()
            if torch.is_tensor(value) and value.ndim >= 2 and value.shape[1] == horizon
        }
        z = mean_center(z, step_params, model_cfg)
        centers.append(z)
    return torch.stack(centers, dim=1)


def rollout_marginal_moments(z0, params, model_cfg):
    """Approximate each recursive JumpRS marginal by propagated first two moments."""
    cfg = model_cfg["model"]
    means = []
    stds = []
    z_mean = z0
    z_var = torch.zeros_like(z0)
    horizon = params["kappa"].shape[1]
    dt = float(cfg["dt"])
    eps_var = float(cfg["eps_var"])
    variance_scale = float(cfg.get("rollout_variance_scale", 1.0))
    for tau in range(horizon):
        step_params = {
            key: value[:, tau]
            for key, value in params.items()
            if torch.is_tensor(value) and value.ndim >= 2 and value.shape[1] == horizon
        }
        log_weights, component_means, component_stds = transition_components(
            step_params, z_mean, model_cfg
        )
        weights = torch.exp(log_weights)
        next_mean = (weights * component_means).sum(dim=-1)
        conditional_var = (
            weights * (component_stds.pow(2) + component_means.pow(2))
        ).sum(dim=-1) - next_mean.pow(2)
        # Euler drift sensitivity propagates uncertainty from the previous state.
        drift_gain = 1.0 - step_params["kappa"] * dt
        next_var = variance_scale ** 2 * conditional_var + drift_gain.pow(2) * z_var
        next_var = torch.clamp(next_var, min=eps_var ** 2)
        z_mean = torch.clamp(next_mean, float(cfg["x_min"]), float(cfg["x_max"]))
        z_var = next_var
        means.append(z_mean)
        stds.append(torch.sqrt(next_var))
    return torch.stack(means, dim=1), torch.stack(stds, dim=1)


def conditioning_path(z0, params, model_cfg):
    cond = []
    z = z0
    horizon = params["kappa"].shape[1]
    for tau in range(horizon):
        cond.append(z)
        step_params = {
            key: value[:, tau]
            for key, value in params.items()
            if torch.is_tensor(value) and value.ndim >= 2 and value.shape[1] == horizon
        }
        z = mean_center(z, step_params, model_cfg)
    return torch.stack(cond, dim=1)


def observed_conditioning_path(z0, observed_x):
    """Teacher-forced chain conditioning path for proper transition scoring."""
    if observed_x.shape[1] == 1:
        return z0.unsqueeze(1)
    return torch.cat([z0.unsqueeze(1), observed_x[:, :-1]], dim=1)
