#!/usr/bin/env bash
set -e
CUDA_VISIBLE_DEVICES=5 python -m baselines.whisper_zeroshot \
  --config configs/l2arctic/whisper_medium_zeroshot_speaker_independent_test_utterance_dev.yaml