#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

bash scripts/run_minimal.sh
(
  cd paper
  pdflatex -interaction=nonstopmode -halt-on-error mainv6.tex
  pdflatex -interaction=nonstopmode -halt-on-error mainv6.tex
)
