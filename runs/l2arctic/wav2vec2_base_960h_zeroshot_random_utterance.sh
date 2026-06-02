#!/usr/bin/env bash
set -e

# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash runs/l2arctic/wav2vec2_base_960h_zeroshot_random_utterance.sh
CUDA_VISIBLE_DEVICES=3 python -m baselines.wav2vec2_ctc_zeroshot \
  --config configs/l2arctic/wav2vec2_base_960h_zeroshot_random_utterance.yaml