# src/data/manifest.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


BASE_REQUIRED_COLUMNS = [
    "utt_id",
    "speaker_id",
    "dataset_name",
    "wav_path",
    "transcript",
]


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)

    records: list[dict[str, Any]] = []

    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {path}:{line_idx}: {e}") from e

    return records


def load_manifest(
    manifest_path: str | Path,
    required_columns: list[str] | None = None,
    validate: bool = True,
) -> pd.DataFrame:
    """
    Load a JSONL ASR manifest.

    By default, this validates only dataset-agnostic ASR fields.
    Dataset-specific fields such as accent_label should be handled by
    experiment configs via evaluation.group_cols, not required globally.
    """
    manifest_path = Path(manifest_path)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    records = read_jsonl(manifest_path)

    if len(records) == 0:
        raise ValueError(f"Manifest is empty: {manifest_path}")

    df = pd.DataFrame(records)

    if validate:
        validate_manifest(df, required_columns=required_columns)

    return df


def validate_manifest(
    df: pd.DataFrame,
    required_columns: list[str] | None = None,
) -> None:
    """
    Validate a manifest dataframe.

    The default required columns are intentionally minimal so that the same
    ASR pipeline can support both L2-ARCTIC and SANDI.

    L2-ARCTIC may have:
        accent_label

    SANDI may have:
        prompt_id
        official_split
        partial_word_count
        hesitation_count

    These should be treated as optional / configurable group columns.
    """
    if required_columns is None:
        required_columns = BASE_REQUIRED_COLUMNS

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns in manifest: {missing}")

    null_required = []

    for col in required_columns:
        if df[col].isna().any():
            null_required.append(col)

    if null_required:
        raise ValueError(f"Required columns contain null values: {null_required}")


def check_manifest_paths(df: pd.DataFrame, max_missing_examples: int = 10) -> None:
    """
    Optional helper: check whether wav_path exists.

    This is not automatically called in load_manifest because path checking
    can be expensive for large datasets.
    """
    if "wav_path" not in df.columns:
        raise ValueError("Cannot check paths because manifest has no wav_path column.")

    missing_paths = []

    for wav_path in df["wav_path"].tolist():
        path = Path(wav_path)
        if not path.exists():
            missing_paths.append(str(path))

    if missing_paths:
        examples = "\n".join(missing_paths[:max_missing_examples])
        raise FileNotFoundError(
            f"Found {len(missing_paths)} missing audio files. "
            f"First examples:\n{examples}"
        )