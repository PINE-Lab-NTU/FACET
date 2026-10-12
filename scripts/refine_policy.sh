#!/usr/bin/env bash
# Contact-weighted refinement of pi_ref (Eq. 7, Alg. 2 line 8) with the VL backbone frozen.
#   PI_REF=<pi_ref>/params CKPT_DIR=/nvme/ckpts bash scripts/refine_policy.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"
XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}" \
$PY -m facet0.refinement.train --dataset "$DATASET" --credit-tag "$ROUND" \
  --base-params "${PI_REF:?}" --exp-name "${EXP_NAME:-$ROUND}" --checkpoint-base-dir "${CKPT_DIR:?}" \
  --contact-weight "${CONTACT_WEIGHT:-1.0}" --wrench-weight "${WRENCH_WEIGHT:-0.1}" \
  --batch-size "${BATCH_SIZE:-32}" --fsdp-devices "${FSDP_DEVICES:-1}" --num-workers "${NUM_WORKERS:-8}" \
  --num-train-steps "${NUM_STEPS:-30000}" "$@"
