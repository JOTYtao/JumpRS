#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
python -m src.run_pvdaq34_rollout_experiments --models Persistence "MC Dropout" PatchTST iTransformer TimesNet "TimeDiff-style" "NsDiff-style" QuantileGRU
