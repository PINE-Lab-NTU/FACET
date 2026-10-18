#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
$PY -m facet0.refinement.encode --dataset "$DATASET" --params "${PI_REF:?}" --out "$FEATURES" \
  --num-proposals "${NUM_PROPOSALS:-1}" --guidance-scale "${GUIDANCE_SCALE:-1.0}" \
  --batch-size "${BATCH_SIZE:-32}" --num-workers "${NUM_WORKERS:-8}"
