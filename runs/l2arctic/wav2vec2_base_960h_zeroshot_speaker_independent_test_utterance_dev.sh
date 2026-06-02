#!/usr/bin/env bash
set -e
CUDA_VISIBLE_DEVICES=0 python -m baselines.wav2vec2_ctc_zeroshot \
  --config configs/l2arctic/wav2vec2_base_960h_zeroshot_speaker_independent_test_utterance_dev.yaml