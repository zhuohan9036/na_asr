from pathlib import Path
import json

import pandas as pd


REQUIRED_COLUMNS = [
    "utt_id",
    "speaker_id",
    "accent_label",
    "dataset_name",
    "wav_path",
    "transcript",
    "split",
]


def load_manifest(path: str) -> pd.DataFrame:
    """
    Load a manifest file.

    Supported formats:
    - .jsonl: one JSON object per line
    - .json: a list of JSON objects
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Manifest file not found: {path}")

    if path.suffix == ".jsonl":
        records = _read_jsonl(path)
    elif path.suffix == ".json":
        records = _read_json(path)
    else:
        raise ValueError(
            f"Unsupported manifest format: {path.suffix}. "
            "Please use .jsonl or .json."
        )

    df = pd.DataFrame(records)
    validate_manifest(df)
    return df


def _read_jsonl(path: Path) -> list[dict]:
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON at line {line_idx} in {path}: {e}"
                ) from e

            if not isinstance(record, dict):
                raise ValueError(
                    f"Each line in a .jsonl manifest must be a JSON object. "
                    f"Line {line_idx} is {type(record).__name__}."
                )

            records.append(record)

    return records


def _read_json(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            f"A .json manifest should be a list of JSON objects, "
            f"but got {type(data).__name__}."
        )

    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(
                f"Every item in a .json manifest should be a JSON object. "
                f"Item {idx} is {type(item).__name__}."
            )

    return data


def validate_manifest(df: pd.DataFrame) -> bool:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns in manifest: {missing}")

    if df.empty:
        raise ValueError("Manifest is empty.")

    null_columns = [
        col for col in REQUIRED_COLUMNS
        if df[col].isnull().any()
    ]

    if null_columns:
        raise ValueError(f"Required columns contain null values: {null_columns}")

    return True


def filter_split(df: pd.DataFrame, split: str) -> pd.DataFrame:
    return df[df["split"] == split].reset_index(drop=True)