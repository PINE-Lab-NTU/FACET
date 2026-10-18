#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
args=()
for c in ${CORPORA:-$DATASET:$FEATURES:$RETURNS_TAG}; do args+=(--corpus "$c"); done
$PY -m facet0.refinement.critic train "${args[@]}" --out "${CRITIC:?}" --steps "${CRITIC_STEPS:-20000}"
