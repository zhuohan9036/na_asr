# scripts/build_metadata_all.py

from pathlib import Path
import json
import random

from src.utils.config import load_config
from src.utils.logging import setup_logger
from src.data.l2arctic import build_l2arctic_metadata


def write_jsonl(records: list[dict], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def assign_speaker_independent_splits(
    records: list[dict],
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
    seed: int = 42,
) -> list[dict]:
    """
    Assign train/dev/test by speaker, not by utterance.

    This avoids putting the same speaker in both train and test.
    """
    rng = random.Random(seed)

    accent_to_speakers = {}

    for record in records:
        accent = record["accent_label"]
        speaker = record["speaker_id"]
        accent_to_speakers.setdefault(accent, set()).add(speaker)

    speaker_to_split = {}

    for accent, speakers in accent_to_speakers.items():
        speakers = sorted(list(speakers))
        rng.shuffle(speakers)

        n = len(speakers)
        n_train = max(1, int(n * train_ratio))
        n_dev = max(1, int(n * dev_ratio)) if n >= 3 else 0

        train_speakers = speakers[:n_train]
        dev_speakers = speakers[n_train:n_train + n_dev]
        test_speakers = speakers[n_train + n_dev:]

        # If test is empty, move one speaker from train to test.
        if len(test_speakers) == 0 and len(train_speakers) > 1:
            test_speakers = [train_speakers.pop()]

        for spk in train_speakers:
            speaker_to_split[spk] = "train"
        for spk in dev_speakers:
            speaker_to_split[spk] = "dev"
        for spk in test_speakers:
            speaker_to_split[spk] = "test"

    new_records = []

    for record in records:
        record = dict(record)
        record["split"] = speaker_to_split[record["speaker_id"]]
        new_records.append(record)

    return new_records


def main():
    config = load_config()

    experiment_name = config["experiment"]["name"]
    logger = setup_logger(experiment_name)

    logger.info("Building metadata_all.jsonl.")
    logger.info(f"Config path: {config['_config_path']}")

    seed = config.get("seed", 42)

    all_records = []

    if config["data"].get("use_l2arctic", True):
        l2_root = config["data"]["l2arctic_root"]
        keep_speakers = config["data"].get("l2arctic_keep_speakers")

        logger.info(f"Building L2-ARCTIC metadata from: {l2_root}")

        l2_records = build_l2arctic_metadata(
            root_dir=l2_root,
            keep_speakers=keep_speakers,
        )

        logger.info(f"L2-ARCTIC records: {len(l2_records)}")
        all_records.extend(l2_records)

    logger.info(f"Total records before split: {len(all_records)}")

    all_records = assign_speaker_independent_splits(
        records=all_records,
        train_ratio=config["split"].get("train_ratio", 0.8),
        dev_ratio=config["split"].get("dev_ratio", 0.1),
        seed=seed,
    )

    output_path = config["output"]["metadata_path"]
    write_jsonl(all_records, output_path)

    logger.info(f"Saved metadata to: {output_path}")
    logger.info(f"Total records saved: {len(all_records)}")

    # Simple summary
    try:
        import pandas as pd

        df = pd.DataFrame(all_records)

        logger.info("Dataset distribution:")
        logger.info(f"\n{df['dataset_name'].value_counts()}")

        logger.info("Accent distribution:")
        logger.info(f"\n{df['accent_label'].value_counts()}")

        logger.info("Speaker distribution by accent:")
        logger.info(f"\n{df.groupby('accent_label')['speaker_id'].nunique()}")

        logger.info("Split distribution:")
        logger.info(f"\n{df['split'].value_counts()}")

        logger.info("Split x accent distribution:")
        logger.info(f"\n{pd.crosstab(df['accent_label'], df['split'])}")

    except Exception as e:
        logger.warning(f"Failed to print summary: {e}")

    logger.info("Finished building metadata.")


if __name__ == "__main__":
    main()