#!/usr/bin/env bash
# Interaction reward of Eq. (4) and the returns sidecar. Judge columns (subgoal_index,
# subgoal_complete, phase) are used when present. ALPHA / BETA / TIME_BUDGET (frames) /
# ENVELOPES (JSON: phase -> 6 per-axis limits) set the exchange rates.
set -euo pipefail
source "$(dirname "$0")/env.sh"
extra=()
[[ -n "${TIME_BUDGET:-}" ]] && extra+=(--time-budget "$TIME_BUDGET")
[[ -n "${ENVELOPES:-}" ]] && extra+=(--envelopes "$ENVELOPES")
$PY -m facet0.refinement.reward --dataset "$DATASET" --tag "$RETURNS_TAG" \
  --gamma "${GAMMA:-0.99}" --alpha "${ALPHA:-0.0}" --beta "${BETA:-0.0}" ${extra[@]+"${extra[@]}"}
