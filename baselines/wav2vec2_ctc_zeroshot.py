# baselines/wav2vec2_ctc_zeroshot.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from src.utils.config import load_config
from src.utils.logging import setup_logger
from src.data.manifest import load_manifest
from src.models.wav2vec2_ctc_asr import Wav2Vec2CTCASR
from src.evaluation.group_metrics import evaluate_asr_predictions, save_metrics


def write_jsonl(records: list[dict[str, Any]], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def get_dataset_name(config: dict) -> str:
    data_config = config.get("data", {})

    if data_config.get("dataset_name"):
        return data_config["dataset_name"]

    test_manifest = str(data_config.get("test_manifest", ""))

    if "data/l2arctic" in test_manifest:
        return "l2arctic"

    if "data/sandi" in test_manifest:
        return "sandi"

    return "general"


def get_optional_fields(row: pd.Series) -> dict[str, Any]:
    optional_fields = [
        "accent_label",
        "native_language",
        "duration",
        "prompt_id",
        "official_split",
        "split",
        "relative_audio_path",
        "partial_word_count",
        "word_count_keep_partial",
        "word_count_drop_partial",
        "hesitation_count",
        "tag_counts",
        "question_text",
        "speaking_time",
    ]

    metadata: dict[str, Any] = {}

    for field in optional_fields:
        if field in row.index:
            value = row[field]

            if pd.isna(value) if not isinstance(value, (dict, list)) else False:
                continue

            metadata[field] = value

    return metadata


def log_column_distribution(
    logger,
    df: pd.DataFrame,
    column: str,
    max_rows: int = 30,
) -> None:
    if column not in df.columns:
        logger.info(f"Column not found, skip distribution: {column}")
        return

    counts = df[column].value_counts()

    logger.info(f"{column} distribution:")
    logger.info(f"\n{counts.head(max_rows)}")

    if len(counts) > max_rows:
        logger.info(f"... truncated. Total unique {column}: {len(counts)}")


def validate_group_cols(df: pd.DataFrame, group_cols: list[str]) -> list[str]:
    return [col for col in group_cols if col in df.columns]


def main() -> None:
    config = load_config()

    experiment_name = config["experiment"]["name"]
    dataset_name = get_dataset_name(config)
    logger = setup_logger(experiment_name, dataset_name=dataset_name)

    logger.info("Starting wav2vec2 CTC zero-shot baseline.")
    logger.info(f"Config path: {config['_config_path']}")
    logger.info(f"Dataset name: {dataset_name}")

    test_manifest = config["data"]["test_manifest"]
    prediction_path = config["output"]["prediction_path"]
    metric_path = config["output"]["metric_path"]

    logger.info(f"Test manifest: {test_manifest}")
    logger.info(f"Prediction path: {prediction_path}")
    logger.info(f"Metric path: {metric_path}")

    test_df = load_manifest(test_manifest)
    logger.info(f"Loaded test samples: {len(test_df)}")
    logger.info(f"Manifest columns: {list(test_df.columns)}")

    evaluation_config = config.get("evaluation", {})
    reference_col = evaluation_config.get("reference_col", "transcript")

    if reference_col not in test_df.columns:
        raise ValueError(
            f"Reference column '{reference_col}' not found in manifest. "
            f"Available columns: {list(test_df.columns)}"
        )

    requested_group_cols = evaluation_config.get("group_cols", ["speaker_id"])
    group_cols = validate_group_cols(test_df, requested_group_cols)
    missing_group_cols = [col for col in requested_group_cols if col not in test_df.columns]

    logger.info(f"Reference column: {reference_col}")
    logger.info(f"Requested group columns: {requested_group_cols}")
    logger.info(f"Effective group columns: {group_cols}")

    if missing_group_cols:
        logger.warning(f"Missing group columns will be ignored: {missing_group_cols}")

    for column in ["dataset_name", "accent_label", "speaker_id", "prompt_id", "official_split"]:
        log_column_distribution(logger, test_df, column)

    model = Wav2Vec2CTCASR(
        model_name=config["model"]["name"],
        device=config["runtime"].get("device", "cuda"),
        sampling_rate=config["model"].get("sampling_rate", 16000),
        torch_dtype=config["model"].get("torch_dtype"),
    )

    logger.info(f"Loaded wav2vec2 CTC model: {config['model']['name']}")
    logger.info(f"Device: {model.device}")
    logger.info(f"Model dtype: {model.model_dtype}")

    prediction_records: list[dict[str, Any]] = []
    failed_records: list[dict[str, Any]] = []

    for _, row in tqdm(
        test_df.iterrows(),
        total=len(test_df),
        desc="wav2vec2 CTC inference",
    ):
        utt_id = row["utt_id"]
        wav_path = row["wav_path"]

        try:
            prediction = model.transcribe_file(wav_path=wav_path)

            record: dict[str, Any] = {
                "utt_id": row["utt_id"],
                "speaker_id": row["speaker_id"],
                "dataset_name": row["dataset_name"],
                "wav_path": row["wav_path"],
                "reference": row[reference_col],
                "reference_col": reference_col,
                "prediction": prediction,
                "model_name": config["model"]["name"],
                "training_setting": "zero_shot",
            }

            record.update(get_optional_fields(row))
            prediction_records.append(record)

        except Exception as e:
            logger.warning(
                f"Failed to transcribe utt_id={utt_id}, wav_path={wav_path}: {e}"
            )

            failed_records.append(
                {
                    "utt_id": utt_id,
                    "wav_path": wav_path,
                    "error": str(e),
                }
            )

    write_jsonl(prediction_records, prediction_path)
    logger.info(f"Saved predictions to: {prediction_path}")
    logger.info(f"Successful predictions: {len(prediction_records)}")
    logger.info(f"Failed predictions: {len(failed_records)}")

    if failed_records:
        failed_path = Path(prediction_path).with_suffix(".failed.jsonl")
        write_jsonl(failed_records, failed_path)
        logger.warning(f"Saved failed records to: {failed_path}")

    if len(prediction_records) == 0:
        logger.error("No successful predictions. Evaluation is skipped.")

        if failed_records:
            logger.error("First failed examples:")
            for item in failed_records[:5]:
                logger.error(json.dumps(item, ensure_ascii=False))

        raise RuntimeError(
            "All inference attempts failed. "
            "Please check the .failed.jsonl file and the log for the actual error."
        )

    pred_df = pd.DataFrame(prediction_records)

    metrics = evaluate_asr_predictions(
        prediction_df=pred_df,
        reference_col="reference",
        prediction_col="prediction",
        group_cols=group_cols,
    )

    metrics["experiment"] = {
        "name": experiment_name,
        "model_name": config["model"]["name"],
        "test_manifest": test_manifest,
        "reference_col": reference_col,
        "group_cols": group_cols,
        "requested_group_cols": requested_group_cols,
    }

    save_metrics(metrics, metric_path)

    logger.info(f"Saved metrics to: {metric_path}")
    logger.info("Overall metrics:")
    logger.info(json.dumps(metrics["overall"], indent=2, ensure_ascii=False))

    logger.info("Finished wav2vec2 CTC zero-shot baseline.")


if __name__ == "__main__":
    main()