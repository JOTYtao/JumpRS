# JumpRS

JumpRS is a research codebase for risk-aware probabilistic photovoltaic (PV)
power and ramp-event forecasting with a regime-aware jump-diffusion state-space
model.

The current experiment uses real measured OEDI PVDAQ AC power only. Synthetic,
simulated, toy, random, or artificial PV power data must not be used for
training, validation, testing, result tables, figures, or paper claims.

## Repository Scope

This GitHub repository is intended to contain source code, configuration,
scripts, tests, and lightweight documentation.

The following local artifacts are intentionally not uploaded:

- real raw or processed datasets under `data/`;
- prediction CSVs and generated outputs under `outputs/`;
- manuscript files under `paper/`;
- large trained model binaries under `artifacts/models/`.

Trained models are saved locally in `artifacts/models/`. Use Git LFS, release
assets, or an external artifact store if model weights need to be distributed.

## Current Experimental Protocol

- Dataset: OEDI PVDAQ systems `4`, `10`, and `34`
- Focused paper experiment: PVDAQ system `34`
- Years: `2011`, `2012`, `2013`
- Resolution: 15 minutes
- History window: 16 steps / 4 hours
- Forecast horizon: 16 recursive rollout steps / 4 hours
- Clear-sky power: site-aligned pvlib workflow with NSRDB weather inputs
- Daylight validity: solar zenith angle below 85 degrees
- Split: first two years for train/validation, third year for frozen test
- Ramp thresholds: `[0.03, 0.05, 0.10, 0.15, 0.20]`

## Model Set

Proposed model:

- `JumpRS`

Benchmarks:

- `Persistence`
- `QuantileGRU`
- `MC Dropout`
- `PatchTST`
- `iTransformer`
- `TimesNet`
- `TimeDiff-style`
- `NsDiff-style`

Benchmark code is organized under `src/benchmarks/<model_name>/`. Shared model
building blocks are kept in `src/models/multistep_baselines.py` so every
benchmark uses the same split, preprocessing, and metric protocol.

## Project Layout

```text
config/                 Data, model, and training configuration
scripts/                Reproducible command-line entry points
src/
  proposed/jumprs/      Proposed JumpRS model wrapper
  benchmarks/           One folder per benchmark family
  data/                 Data loading, clear-sky construction, preprocessing
  features/             Ramp labels and forecasting windows
  models/               Core PyTorch model implementations
  training/             JumpRS training and losses
  evaluation/           Metrics
  inference/            Prediction export from trained artifacts
  visualization/        Paper/result figures
artifacts/models/       Local trained model checkpoints, ignored by Git
data/                   Local raw/processed real data, ignored by Git
outputs/                Local predictions/metrics/figures, ignored by Git
tests/                  Unit tests
```

## Environment

Using `pip`:

```bash
cd /path/to/JumpRS
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Using conda:

```bash
conda env create -f environment.yml
conda activate jumprs
```

The main dependencies are Python, NumPy, pandas, PyYAML, scikit-learn,
matplotlib, pvlib, PyTorch, pyarrow, and pytest.

## Data Preparation

The data pipeline requires real measured PVDAQ power and NSRDB weather. Configure
site metadata and columns in `config/data.yaml`.

Download PVDAQ power:

```bash
python scripts/download_pvdaq_multisite.py
```

Download NSRDB weather for clear-sky power construction:

```bash
export NSRDB_API_KEY="<your NREL developer API key>"
export NSRDB_EMAIL="<your NSRDB account email>"
python scripts/download_nsrdb_weather.py
```

Preprocess data and build train/validation/test windows:

```bash
bash scripts/prepare_multisite.sh
```

Preparation writes local files such as:

```text
data/processed/multisite/<site_id>/processed_power.csv
outputs/prepared/multisite_splits.npz
```

These files are not uploaded to GitHub.

## Train JumpRS

Train only the proposed JumpRS model on the current PVDAQ 34 four-hour rollout
task:

```bash
bash scripts/train_jumprs.sh
```

Equivalent explicit command:

```bash
python -m src.run_pvdaq34_rollout_experiments --models JumpRS
```

The trained checkpoint and metadata are written locally to:

```text
artifacts/models/pvdaq_system_34_rollout_4h/jumprs/
```

## Train Benchmarks

Train all retained benchmarks:

```bash
bash scripts/train_benchmarks.sh
```

Equivalent explicit command:

```bash
python -m src.run_pvdaq34_rollout_experiments \
  --models Persistence "MC Dropout" PatchTST iTransformer TimesNet \
           "TimeDiff-style" "NsDiff-style" QuantileGRU
```

Each trained benchmark is saved under:

```text
artifacts/models/pvdaq_system_34_rollout_4h/<model_slug>/
```

Run the full PVDAQ 34 comparison in one command:

```bash
python -m src.run_pvdaq34_rollout_experiments --models all
```

## Inference From Trained Models

After checkpoints exist in `artifacts/models/`, export prediction CSVs:

```bash
bash scripts/infer_from_artifacts.sh \
  --run-name pvdaq_system_34_rollout_4h \
  --site-id pvdaq_system_34 \
  --model all
```

To export a single model:

```bash
bash scripts/infer_from_artifacts.sh --model JumpRS
```

Prediction files are written to `outputs/predictions/` and are ignored by Git.
Each lead-time file contains target time, actual power, point prediction, and
100 predictive samples.

## Metrics and Figures

Compute PVDAQ 34 lead-wise metrics from prediction CSVs:

```bash
python -m src.evaluation.compute_pvdaq34_lead_metrics
```

Generate current result figures:

```bash
python -m src.visualization.paper_figures
```

Metrics and figures are local generated outputs and are not uploaded by default.

## Tests

Run the focused unit test suite:

```bash
python -m pytest
```

Synthetic data is allowed only inside unit tests for code-shape and numerical
stability checks.

## GitHub Upload

This repository is prepared for:

```bash
git remote add origin git@github.com:JOTYtao/JumpRS.git
git push -u origin main
```

Only code, configuration, scripts, tests, and lightweight documentation should
be committed. Keep real datasets, prediction archives, manuscript files, and
large model binaries out of normal Git history.
