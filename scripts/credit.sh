#!/usr/bin/env bash
# V_psi for every frame, then contact-selective credit (Eq. 6) and labels b_t.
set -euo pipefail
source "$(dirname "$0")/env.sh"
$PY -m facet0.refinement.critic score --critic "${CRITIC:?}" --corpus "$DATASET:$FEATURES:$RETURNS_TAG" \
  --out "$DATASET/meta/values_$ROUND.parquet"
$PY -m facet0.refinement.credit --dataset "$DATASET" --tag "$ROUND" \
  --scores "$DATASET/meta/values_$ROUND.parquet" --returns-tag "$RETURNS_TAG" \
  --horizon "${CREDIT_HORIZON:-20}" --positive-fraction "${POSITIVE_FRACTION:-0.2}"
