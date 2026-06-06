#!/usr/bin/env bash
set -e

# Usage:
#   bash runs/sandi/build_metadata_all.sh

python -m scripts.sandi.build_metadata_all \
  --config configs/sandi/build_metadata_all.yaml