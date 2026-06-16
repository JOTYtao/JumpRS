#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python -m src.run_multisite_experiments
python -m src.visualization.paper_figures
