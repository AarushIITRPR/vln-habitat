#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"

CUDA_VISIBLE_DEVICES= \
PYTHONPATH="../habitat-baselines:../habitat-lab:." \
"${PYTHON_BIN}" \
  scripts/current_habitat_cma_da_train.py
