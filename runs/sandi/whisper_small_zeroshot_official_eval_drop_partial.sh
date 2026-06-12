#!/usr/bin/env bash
set -e

CUDA_VISIBLE_DEVICES=0 python -m baselines.whisper_zeroshot \
  --config configs/sandi/whisper_small_zeroshot_official_eval_drop_partial.yaml