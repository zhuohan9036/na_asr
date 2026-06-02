#!/usr/bin/env bash
set -e
CUDA_VISIBLE_DEVICES=2 python -m baselines.whisper_zeroshot \
  --config configs/l2arctic/whisper_large_v3_zeroshot_random_utterance.yaml