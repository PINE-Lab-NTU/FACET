#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
extra=()
[[ -n "${TIME_BUDGET:-}" ]] && extra+=(--time-budget "$TIME_BUDGET")
[[ -n "${ENVELOPES:-}" ]] && extra+=(--envelopes "$ENVELOPES")
$PY -m facet0.refinement.reward --dataset "$DATASET" --tag "$RETURNS_TAG" \
  --gamma "${GAMMA:-0.99}" --alpha "${ALPHA:?}" --beta "${BETA:?}" ${extra[@]+"${extra[@]}"}
