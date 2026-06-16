# JumpRS Project Brief

## Objective

Develop and evaluate JumpRS, a regime-aware jump-diffusion state-space model
for probabilistic PV trajectory and ramp-event forecasting using real measured
solar power data.

## Active Dataset Protocol

- Source: OEDI/PVDAQ systems `4`, `10`, and `34`
- Focused current experiment: PVDAQ system `34`
- Years: `2011`, `2012`, `2013`
- Resolution: 15 minutes
- Clear-sky power: site-aligned `pvlib` workflow using NSRDB air temperature
  and wind speed in the Faiman/PVWatts conversion
- Daylight validity: solar zenith angle below `85` degrees
- Split: first two years for training/validation; third year for testing
- Validation: final 15% of the first two-year period
- History: 16 steps / 4 hours
- Forecast: 16 recursive rollout steps / 4 hours

Only real measured PV power may be used for experimental results. NSRDB weather
must cover every daylight-valid timestamp used to calculate clear-sky power.
Synthetic data is permitted only in isolated unit tests.

## Ramp Definition

For site capacity `P_cap`, ramp window `L_r`, and threshold `gamma`:

```text
rho_t = (P_t - P_{t-Lr}) / P_cap
down_t = 1{rho_t <= -gamma}
up_t   = 1{rho_t >=  gamma}
```

Active thresholds are `[0.03, 0.05, 0.10, 0.15, 0.20]`. All labels and metrics
must use the daylight-valid mask.

## Current JumpRS Core

JumpRS models clear-sky-normalized PV power with:

- context-dependent mean reversion;
- constant continuous diffusion loading;
- signed compound Poisson jumps;
- soft regime mixing;
- innovation-driven posterior belief filtering;
- recursive Monte Carlo rollout for trajectory distributions and ramp-event
  probabilities.

The current training objective contains exactly two terms:

```text
loss = transition_nll_weight * transition_nll
     + mixture_crps_weight * mixture_crps
```

No auxiliary point, quantile, event-classification, trajectory-MSE, smoothness,
or parameter-regularization loss is used in the retained JumpRS objective.

## Retained Benchmarks

- Persistence
- QuantileGRU
- MC Dropout
- PatchTST
- iTransformer
- TimesNet
- TimeDiff-style
- NsDiff-style

## Required Evaluation

Deterministic metrics:

- MAE
- RMSE
- nRMSE

Probabilistic metrics:

- CRPS
- CRPSS
- Brier Score
- AUPRC

All metrics are computed by lead time for the 16-step rollout horizon.

## Repository Policy

The public GitHub repository should include code, configuration, scripts, tests,
and lightweight documentation. Real datasets, prediction CSV archives,
manuscript files, and large trained model binaries are local artifacts and are
not uploaded through ordinary Git.
