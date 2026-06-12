#!/usr/bin/env bash
set -e

CUDA_VISIBLE_DEVICES=1 python -m baselines.whisper_zeroshot \
  --config configs/sandi/whisper_medium_zeroshot_official_dev_drop_partial.yaml