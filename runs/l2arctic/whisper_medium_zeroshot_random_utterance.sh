#!/usr/bin/env bash
set -e
CUDA_VISIBLE_DEVICES=1 python -m baselines.whisper_zeroshot \
  --config configs/l2arctic/whisper_medium_zeroshot_random_utterance.yaml