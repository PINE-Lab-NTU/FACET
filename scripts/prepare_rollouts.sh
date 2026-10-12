#!/usr/bin/env bash
# Deployment rollouts of pi_ref -> RL-ready corpus, then per-axis normalization statistics.
#   RAW=<rollout export> DATASET=<corpus> ROUND=r1 OPENPI_SRC=... bash scripts/prepare_rollouts.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"
$PY -m facet0.data.rollouts --src "${RAW:?}" --dst "$DATASET"
$PY -m facet0.data.norm_stats --dataset "$DATASET"
