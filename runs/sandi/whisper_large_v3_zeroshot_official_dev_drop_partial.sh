#!/usr/bin/env bash
set -e

CUDA_VISIBLE_DEVICES=4 python -m baselines.whisper_zeroshot \
  --config configs/sandi/whisper_large_v3_zeroshot_official_dev_drop_partial.yaml