# scripts/sandi/build_metadata_all.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.data.sandi import SANDI_SPLITS, build_sandi_metadata_for_split
from src.utils.config import load_config
from src.utils.logging import setup_logger


def write_jsonl(records: list[dict[str, Any]], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(obj: dict[str, Any], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def main() -> None:
    config = load_config()

    experiment_name = config["experiment"]["name"]
    logger = setup_logger(experiment_name)

    sandi_root = Path(config["data"]["sandi_root"]).resolve()
    dataset_name = config["data"].get("dataset_name", "SpeakAndImprove2025")
    strict = bool(config["data"].get("strict", False))

    output_metadata_path = Path(config["output"]["metadata_path"])
    manifest_dir = Path(config["output"]["manifest_dir"])
    summary_path = Path(config["output"]["summary_path"])

    logger.info("Building Speak & Improve / SANDI metadata.")
    logger.info(f"Config path: {config['_config_path']}")
    logger.info(f"SANDI root: {sandi_root}")
    logger.info(f"Strict mode: {strict}")

    all_records: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}

    for split in SANDI_SPLITS:
        logger.info(f"Processing official split: {split}")

        records, summary = build_sandi_metadata_for_split(
            sandi_root=sandi_root,
            split=split,
            dataset_name=dataset_name,
            strict=strict,
        )

        all_records.extend(records)
        summaries[split] = summary

        official_manifest_path = manifest_dir / f"official_{split}.jsonl"
        write_jsonl(records, official_manifest_path)

        logger.info(f"Saved official {split} manifest: {official_manifest_path}")
        logger.info(json.dumps(summary, indent=2, ensure_ascii=False))

    write_jsonl(all_records, output_metadata_path)
    logger.info(f"Saved metadata_all: {output_metadata_path}")
    logger.info(f"Total records: {len(all_records)}")

    # Dataset-level summary.
    dataset_summary: dict[str, Any] = {
        "dataset_name": dataset_name,
        "sandi_root": str(sandi_root),
        "total_records": len(all_records),
        "official_split_summaries": summaries,
    }

    try:
        import pandas as pd

        df = pd.DataFrame(all_records)

        dataset_summary["official_split_counts"] = (
            df["official_split"].value_counts().sort_index().to_dict()
        )
        dataset_summary["num_speakers"] = int(df["speaker_id"].nunique())
        dataset_summary["num_prompts"] = int(df["prompt_id"].nunique())
        dataset_summary["utterances_per_speaker_summary"] = (
            df.groupby("speaker_id").size().describe().to_dict()
        )
        dataset_summary["utterances_per_prompt"] = (
            df["prompt_id"].value_counts().sort_index().to_dict()
        )
        dataset_summary["partial_word_count_total"] = int(df["partial_word_count"].sum())
        dataset_summary["hesitation_count_total"] = int(df["hesitation_count"].sum())

        logger.info("Official split counts:")
        logger.info(f"\n{df['official_split'].value_counts().sort_index()}")

        logger.info("Utterances per speaker summary:")
        logger.info(f"\n{df.groupby('speaker_id').size().describe()}")

        logger.info("Prompt distribution:")
        logger.info(f"\n{df['prompt_id'].value_counts().sort_index()}")

    except Exception as e:
        logger.warning(f"Failed to compute pandas summary: {e}")

    write_json(dataset_summary, summary_path)
    logger.info(f"Saved summary: {summary_path}")
    logger.info("Finished building SANDI metadata.")


if __name__ == "__main__":
    main()