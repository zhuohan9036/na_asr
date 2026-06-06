#!/usr/bin/env bash
set -e

# Usage:
#   bash runs/sandi/make_asr_splits.sh

python -m scripts.sandi.make_asr_splits \
  --config configs/sandi/make_asr_splits.yaml