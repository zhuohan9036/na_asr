# scripts/make_asr_splits.py

import json
from pathlib import Path

from src.utils.config import load_config
from src.utils.logging import setup_logger
from src.data.split_manifest import (
    read_jsonl,
    write_split_manifests,
    make_random_utterance_split,
    make_speaker_independent_split,
    make_speaker_independent_test_utterance_dev_split,
    summarize_split,
    save_summary,
)


def main():
    config = load_config()

    experiment_name = config["experiment"]["name"]
    logger = setup_logger(experiment_name)

    logger.info("Making ASR splits.")
    logger.info(f"Config path: {config['_config_path']}")

    input_metadata = config["data"]["input_metadata"]
    output_dir = config["output"]["manifest_dir"]
    summary_path = config["output"].get("summary_path")

    strategy = config["split"]["strategy"]
    seed = config.get("seed", 42)

    logger.info(f"Input metadata: {input_metadata}")
    logger.info(f"Output manifest dir: {output_dir}")
    logger.info(f"Split strategy: {strategy}")
    logger.info(f"Seed: {seed}")

    records = read_jsonl(input_metadata)
    logger.info(f"Loaded records: {len(records)}")

    if strategy == "random_utterance":
        split_records = make_random_utterance_split(
            records=records,
            train_ratio=config["split"]["train_ratio"],
            dev_ratio=config["split"]["dev_ratio"],
            test_ratio=config["split"]["test_ratio"],
            seed=seed,
            stratify_by_accent=config["split"].get("stratify_by_accent", True),
        )

    elif strategy == "speaker_independent":
        split_records = make_speaker_independent_split(
            records=records,
            train_ratio=config["split"]["train_ratio"],
            dev_ratio=config["split"]["dev_ratio"],
            seed=seed,
        )

    elif strategy == "speaker_independent_test_utterance_dev":
        split_records = make_speaker_independent_test_utterance_dev_split(
            records=records,
            test_speaker_ratio=config["split"]["test_speaker_ratio"],
            dev_utterance_ratio=config["split"]["dev_utterance_ratio"],
            seed=seed,
        )

    else:
        raise ValueError(f"Unsupported split strategy: {strategy}")

    write_split_manifests(split_records, output_dir)
    logger.info(f"Saved split manifests to: {output_dir}")

    summary = summarize_split(split_records)

    logger.info("Split counts:")
    logger.info(json.dumps(summary["split_counts"], indent=2, ensure_ascii=False))

    logger.info("Speaker counts by split:")
    logger.info(json.dumps(summary["speaker_counts_by_split"], indent=2, ensure_ascii=False))

    logger.info("Utterance counts by accent and split:")
    logger.info(json.dumps(summary["utterance_counts_by_accent_and_split"], indent=2, ensure_ascii=False))

    logger.info("Speaker counts by accent and split:")
    logger.info(json.dumps(summary["speaker_counts_by_accent_and_split"], indent=2, ensure_ascii=False))

    if summary_path is not None:
        save_summary(summary, summary_path)
        logger.info(f"Saved split summary to: {summary_path}")

    logger.info("Finished making ASR splits.")


if __name__ == "__main__":
    main()