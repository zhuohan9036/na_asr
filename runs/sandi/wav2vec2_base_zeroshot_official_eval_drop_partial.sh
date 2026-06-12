
CUDA_VISIBLE_DEVICES=0 python -m baselines.wav2vec2_ctc_zeroshot \
  --config configs/sandi/wav2vec2_base_zeroshot_official_eval_drop_partial.yaml
