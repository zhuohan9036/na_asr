#!/usr/bin/env bash
set -e
CUDA_VISIBLE_DEVICES=6 python -m baselines.whisper_zeroshot \
  --config configs/l2arctic/whisper_small_zeroshot_speaker_independent_test_utterance_dev.yaml