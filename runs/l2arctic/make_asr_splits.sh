python -m scripts.l2arctic.make_asr_splits --config configs/l2arctic/split_speaker_independent.yaml
python -m scripts.l2arctic.make_asr_splits --config configs/l2arctic/split_random_utterance.yaml
python -m scripts.l2arctic.make_asr_splits --config configs/l2arctic/split_speaker_independent_test_utterance_dev.yaml