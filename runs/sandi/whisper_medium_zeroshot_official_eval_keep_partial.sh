#!/usr/bin/env bash
set -e

CUDA_VISIBLE_DEVICES=7 python -m baselines.whisper_zeroshot \
  --config configs/sandi/whisper_medium_zeroshot_official_eval_keep_partial.yaml