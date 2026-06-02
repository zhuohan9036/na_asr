#!/usr/bin/env bash
set -e
CUDA_VISIBLE_DEVICES=4 python -m baselines.wav2vec2_ctc_zeroshot \
  --config configs/l2arctic/wav2vec2_large_960h_lv60_self_zeroshot_random_utterance.yaml