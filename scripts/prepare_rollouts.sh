#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
$PY -m facet0.data.rollouts --src "${RAW:?}" --dst "$DATASET"
