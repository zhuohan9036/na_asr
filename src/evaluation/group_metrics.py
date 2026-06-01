# src/evaluation/group_metrics.py

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.evaluation.asr_metrics import compute_wer, compute_cer


def evaluate_asr_predictions(
    prediction_df: pd.DataFrame,
    reference_col: str = "reference",
    prediction_col: str = "prediction",
    group_cols: list[str] | None = None,
) -> dict:
    if group_cols is None:
        group_cols = ["accent_label", "speaker_id"]

    df = prediction_df.copy()

    df["wer"] = df.apply(
        lambda row: compute_wer(row[reference_col], row[prediction_col]),
        axis=1,
    )

    df["cer"] = df.apply(
        lambda row: compute_cer(row[reference_col], row[prediction_col]),
        axis=1,
    )

    metrics = {
        "overall": {
            "utterance_count": int(len(df)),
            "mean_wer": float(df["wer"].mean()),
            "mean_cer": float(df["cer"].mean()),
        },
        "by_group": {},
    }

    for group_col in group_cols:
        group_result = {}

        for group_name, group_df in df.groupby(group_col):
            group_result[str(group_name)] = {
                "utterance_count": int(len(group_df)),
                "mean_wer": float(group_df["wer"].mean()),
                "mean_cer": float(group_df["cer"].mean()),
            }

        metrics["by_group"][group_col] = group_result

    return metrics


def save_metrics(metrics: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)