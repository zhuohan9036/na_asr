# scripts/build_metadata_all.py

from pathlib import Path
import json

from src.utils.config import load_config
from src.utils.logging import setup_logger
from src.data.l2arctic import build_l2arctic_metadata


def write_jsonl(records: list[dict], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            # Ensure metadata_all.jsonl does not contain split information.
            record = dict(record)
            record.pop("split", None)

            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    config = load_config()

    experiment_name = config["experiment"]["name"]
    logger = setup_logger(experiment_name)

    logger.info("Building metadata_all.jsonl.")
    logger.info(f"Config path: {config['_config_path']}")

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

    logger.info(f"Total records before saving: {len(all_records)}")

    # Remove split information from every record, just in case the dataset builder added it.
    clean_records = []
    for record in all_records:
        record = dict(record)
        record.pop("split", None)
        clean_records.append(record)

    output_path = config["output"]["metadata_path"]
    write_jsonl(clean_records, output_path)

    logger.info(f"Saved metadata to: {output_path}")
    logger.info(f"Total records saved: {len(clean_records)}")

    # Simple summary
    try:
        import pandas as pd

        df = pd.DataFrame(clean_records)

        logger.info("Dataset distribution:")
        logger.info(f"\n{df['dataset_name'].value_counts()}")

        logger.info("Accent distribution:")
        logger.info(f"\n{df['accent_label'].value_counts()}")

        logger.info("Speaker distribution by accent:")
        logger.info(f"\n{df.groupby('accent_label')['speaker_id'].nunique()}")

        logger.info("Utterance distribution by accent:")
        logger.info(f"\n{df.groupby('accent_label')['utt_id'].count()}")

        if "split" in df.columns:
            logger.warning("Unexpected split column found in metadata.")
        else:
            logger.info("No split column in metadata. This is expected.")

    except Exception as e:
        logger.warning(f"Failed to print summary: {e}")

    logger.info("Finished building metadata.")


if __name__ == "__main__":
    main()