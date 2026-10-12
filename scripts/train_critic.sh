#!/usr/bin/env bash
# Action-Wrench Critic on the accumulated rollout set D.
#   CORPORA="d1:d1/meta/features_r2:r1 d2:d2/meta/features_r2:r2" CRITIC=critics/r2 bash scripts/train_critic.sh
# CORPORA defaults to this round's corpus alone.
set -euo pipefail
source "$(dirname "$0")/env.sh"
args=()
for c in ${CORPORA:-$DATASET:$FEATURES:$RETURNS_TAG}; do args+=(--corpus "$c"); done
$PY -m facet0.refinement.critic train "${args[@]}" --out "${CRITIC:?}" --steps "${CRITIC_STEPS:-20000}"
