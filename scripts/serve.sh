#!/usr/bin/env bash
#   OPENPI_SRC=<openpi>/src MODEL_DIR=<checkpoint step dir> bash scripts/serve.sh
set -euo pipefail
export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd):${OPENPI_SRC:?}:${PYTHONPATH:-}"
${PY:-python} -m facet0.policy.serve --model-dir "${MODEL_DIR:?}" --port "${PORT:-8000}" --guidance-scale "${GUIDANCE_SCALE:-1.0}"
