# Project Structure and Retained Artifacts

The repository keeps artifacts needed for reproducibility, paper verification, or ongoing research. Regenerable caches, obsolete single-site MVP artifacts, superseded manuscript backups, and large per-model intermediate prediction files are removed.

## Data

- `data/raw/multisite/`: authoritative real measured PVDAQ source files.
- `data/raw/nsrdb/`: NSRDB PSM3 weather inputs used for site-aligned clear-sky construction.
- `data/processed/multisite/`: current site-level 15-minute processed data.
- `outputs/prepared/multisite_splits.npz`: current training, validation, and test windows.

The earlier single-site 2022/2023 MVP data was removed after the project moved to the three-site 2011-2013 protocol.

## Research State

- `research-state.yaml`: current project phase, constraints, and next actions.
- `findings.md`: current synthesis of what is known, constraints, and open questions.
- `research-log.md`: chronological experiment and paper-revision log.

The former `research/` duplicate state directory was removed because it lagged behind the root-level state files and created conflicting project status.

## Metrics and Predictions

- Unprefixed multisite CSVs: primary baseline and main-run results.
- Historical `h*`, obsolete `raw_*`, tuning, optimization, and old JumpRS candidate CSVs are not retained in the cleaned current project state; paper claims should use the current rerun outputs and the paper-revision tables.
- `outputs/predictions/raw_test_sample_predictions_multisite.csv`: retained unified test-sample prediction archive.

Individual H-series prediction files and the superseded calibrated merged prediction file were removed. They can be regenerated from experiment scripts if needed.

## Paper

The active manuscript is `paper/mainv6.tex`, with compiled output at `paper/mainv6.pdf`. Old manuscript versions, LaTeX build caches, legacy tables, and the obsolete single-site manuscript were removed.

Current paper figures and paper-revision tables are read from `outputs/figures/` and `outputs/paper_revision/`, so these directories are retained.

## Reproducible Entry Points

- `scripts/download_pvdaq_multisite.py`
- `scripts/prepare_multisite.sh`
- `scripts/run_minimal.sh`
- `scripts/run_all.sh`
