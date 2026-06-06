# baselines/wav2vec2_ctc_zeroshot.py

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.utils.config import load_config
from src.utils.logging import setup_logger
from src.data.manifest import load_manifest
from src.models.wav2vec2_ctc_asr import Wav2Vec2CTCASR
from src.evaluation.group_metrics import evaluate_asr_predictions, save_metrics


def write_jsonl(records: list[dict], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    config = load_config()

    experiment_name = config["experiment"]["name"]
    logger = setup_logger(experiment_name)

    logger.info("Starting wav2vec2 CTC zero-shot baseline.")
    logger.info(f"Config path: {config['_config_path']}")

    test_manifest = config["data"]["test_manifest"]
    prediction_path = config["output"]["prediction_path"]
    metric_path = config["output"]["metric_path"]

    logger.info(f"Test manifest: {test_manifest}")
    logger.info(f"Prediction path: {prediction_path}")
    logger.info(f"Metric path: {metric_path}")

    test_df = load_manifest(test_manifest)
    logger.info(f"Loaded test samples: {len(test_df)}")

    reference_col = config.get("evaluation", {}).get("reference_col", "transcript")

    if reference_col not in test_df.columns:
        raise ValueError(
            f"Reference column '{reference_col}' not found in manifest. "
            f"Available columns: {list(test_df.columns)}"
        )

    logger.info(f"Reference column: {reference_col}")

    logger.info("Accent distribution:")
    logger.info(f"\n{test_df['accent_label'].value_counts()}")

    logger.info("Speaker distribution:")
    logger.info(f"\n{test_df['speaker_id'].value_counts()}")

    model = Wav2Vec2CTCASR(
        model_name=config["model"]["name"],
        device=config["runtime"].get("device", "cuda"),
        sampling_rate=config["model"].get("sampling_rate", 16000),
        torch_dtype=config["model"].get("torch_dtype"),
    )

    logger.info(f"Loaded wav2vec2 CTC model: {config['model']['name']}")
    logger.info(f"Device: {model.device}")
    logger.info(f"Model dtype: {model.model_dtype}")

    prediction_records = []
    failed_records = []

    for _, row in tqdm(
        test_df.iterrows(),
        total=len(test_df),
        desc="wav2vec2 CTC inference",
    ):
        utt_id = row["utt_id"]
        wav_path = row["wav_path"]

        try:
            prediction = model.transcribe_file(wav_path=wav_path)

            # record = {
            #     "utt_id": row["utt_id"],
            #     "speaker_id": row["speaker_id"],
            #     "accent_label": row["accent_label"],
            #     "dataset_name": row["dataset_name"],
            #     "wav_path": row["wav_path"],
            #     "reference": row["transcript"],
            #     "prediction": prediction,
            #     "model_name": config["model"]["name"],
            #     "training_setting": "zero_shot",
            # }
            record = {
                "utt_id": row["utt_id"],
                "speaker_id": row["speaker_id"],
                "accent_label": row["accent_label"] if "accent_label" in row else row.get("dataset_name", "unknown"),
                "dataset_name": row["dataset_name"],
                "wav_path": row["wav_path"],
                "reference": row[reference_col],
                "reference_col": reference_col,
                "prediction": prediction,
                "model_name": config["model"]["name"],
                "training_setting": "zero_shot",
            }

            if "duration" in row:
                record["duration"] = row["duration"]

            if "native_language" in row:
                record["native_language"] = row["native_language"]

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

    # metrics = evaluate_asr_predictions(
    #     prediction_df=pred_df,
    #     reference_col="reference",
    #     prediction_col="prediction",
    #     group_cols=["accent_label", "speaker_id"],
    # )

    group_cols = config.get("evaluation", {}).get("group_cols", ["speaker_id"])

    metrics = evaluate_asr_predictions(
        prediction_df=pred_df,
        reference_col="reference",
        prediction_col="prediction",
        group_cols=group_cols,
    )

    save_metrics(metrics, metric_path)

    logger.info(f"Saved metrics to: {metric_path}")
    logger.info("Overall metrics:")
    logger.info(json.dumps(metrics["overall"], indent=2, ensure_ascii=False))

    logger.info("Finished wav2vec2 CTC zero-shot baseline.")


if __name__ == "__main__":
    main()