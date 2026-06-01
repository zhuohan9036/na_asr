python -m scripts.make_asr_splits --config configs/split_speaker_independent.yaml
python -m scripts.make_asr_splits --config configs/split_random_utterance.yaml
python -m scripts.make_asr_splits --config configs/split_speaker_independent_test_utterance_dev.yaml