# src/evaluation/group_metrics.py

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.evaluation.asr_metrics import (
    compute_wer_stats,
    compute_cer_stats,
    compute_corpus_error_rate,
)


def _add_utterance_metrics(
    df: pd.DataFrame,
    reference_col: str,
    prediction_col: str,
) -> pd.DataFrame:
    """
    Add utterance-level WER/CER and edit-distance statistics.
    """
    df = df.copy()

    wer_stats_list = []
    cer_stats_list = []

    for _, row in df.iterrows():
        reference = row[reference_col]
        prediction = row[prediction_col]

        wer_stats = compute_wer_stats(reference, prediction)
        cer_stats = compute_cer_stats(reference, prediction)

        wer_stats_list.append(wer_stats)
        cer_stats_list.append(cer_stats)

    df["wer_errors"] = [item.errors for item in wer_stats_list]
    df["wer_ref_words"] = [item.reference_length for item in wer_stats_list]
    df["wer"] = [item.error_rate for item in wer_stats_list]

    df["cer_errors"] = [item.errors for item in cer_stats_list]
    df["cer_ref_chars"] = [item.reference_length for item in cer_stats_list]
    df["cer"] = [item.error_rate for item in cer_stats_list]

    return df


def _summarize_metric_df(df: pd.DataFrame) -> dict:
    """
    Summarize one dataframe into mean and corpus-level WER/CER.
    """
    total_wer_errors = int(df["wer_errors"].sum())
    total_wer_ref_words = int(df["wer_ref_words"].sum())

    total_cer_errors = int(df["cer_errors"].sum())
    total_cer_ref_chars = int(df["cer_ref_chars"].sum())

    if total_wer_ref_words == 0:
        corpus_wer = 0.0 if total_wer_errors == 0 else 1.0
    else:
        corpus_wer = total_wer_errors / total_wer_ref_words

    if total_cer_ref_chars == 0:
        corpus_cer = 0.0 if total_cer_errors == 0 else 1.0
    else:
        corpus_cer = total_cer_errors / total_cer_ref_chars

    return {
        "utterance_count": int(len(df)),

        # Utterance-level average.
        "mean_wer": float(df["wer"].mean()),
        "mean_cer": float(df["cer"].mean()),

        # Corpus-level error rate.
        "corpus_wer": float(corpus_wer),
        "corpus_cer": float(corpus_cer),

        # Raw counts, useful for debugging and reporting.
        "total_wer_errors": total_wer_errors,
        "total_wer_ref_words": total_wer_ref_words,
        "total_cer_errors": total_cer_errors,
        "total_cer_ref_chars": total_cer_ref_chars,
    }


def evaluate_asr_predictions(
    prediction_df: pd.DataFrame,
    reference_col: str = "reference",
    prediction_col: str = "prediction",
    group_cols: list[str] | None = None,
) -> dict:
    """
    Evaluate ASR predictions.

    Returns both:
    - mean_wer / mean_cer:
        average of utterance-level WER/CER
    - corpus_wer / corpus_cer:
        total edit errors divided by total reference length

    In ASR research, corpus_wer is usually the more standard main metric.
    """
    if group_cols is None:
        group_cols = ["accent_label", "speaker_id"]

    required_cols = [reference_col, prediction_col] + group_cols
    missing = [col for col in required_cols if col not in prediction_df.columns]

    if missing:
        raise ValueError(f"Missing required columns for evaluation: {missing}")

    df = _add_utterance_metrics(
        df=prediction_df,
        reference_col=reference_col,
        prediction_col=prediction_col,
    )

    metrics = {
        "overall": _summarize_metric_df(df),
        "by_group": {},
    }

    for group_col in group_cols:
        group_result = {}

        for group_name, group_df in df.groupby(group_col):
            group_result[str(group_name)] = _summarize_metric_df(group_df)

        metrics["by_group"][group_col] = group_result

    return metrics


def save_metrics(metrics: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)