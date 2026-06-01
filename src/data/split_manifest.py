# src/data/split_manifest.py

from __future__ import annotations

import json
import random
from pathlib import Path
from collections import defaultdict
from typing import Any

import pandas as pd


REQUIRED_METADATA_COLUMNS = [
    "utt_id",
    "speaker_id",
    "accent_label",
    "dataset_name",
    "wav_path",
    "transcript",
]


VALID_SPLITS = {"train", "dev", "test"}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON at line {line_idx} in {path}: {e}"
                ) from e

            if not isinstance(item, dict):
                raise ValueError(
                    f"Line {line_idx} in {path} is not a JSON object."
                )

            records.append(item)

    validate_unsplit_metadata(records)
    return records


def validate_unsplit_metadata(records: list[dict[str, Any]]) -> None:
    """
    Validate metadata_all.jsonl.

    This function assumes metadata_all.jsonl is the clean, unsplit source file.
    Therefore:
    - required metadata columns must exist
    - split must NOT exist
    """
    if not records:
        raise ValueError("No records found in metadata.")

    for idx, item in enumerate(records):
        missing = [col for col in REQUIRED_METADATA_COLUMNS if col not in item]

        if missing:
            raise ValueError(
                f"Record {idx} is missing required metadata columns: {missing}"
            )

        if "split" in item:
            raise ValueError(
                "Input metadata already contains a 'split' field. "
                "metadata_all.jsonl should be unsplit. "
                "Please rebuild metadata_all.jsonl without split, or remove the field first."
            )


def write_jsonl(records: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for item in records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def write_split_manifests(
    records: list[dict[str, Any]],
    output_dir: str | Path,
) -> None:
    """
    Write:
    - train.jsonl
    - dev.jsonl
    - test.jsonl
    - all.jsonl

    The input records must already contain a valid split field.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_to_records = {
        "train": [],
        "dev": [],
        "test": [],
    }

    for idx, item in enumerate(records):
        if "split" not in item:
            raise ValueError(f"Record {idx} does not contain a split field.")

        split = item["split"]

        if split not in VALID_SPLITS:
            raise ValueError(f"Unknown split: {split}")

        split_to_records[split].append(item)

    for split, split_records in split_to_records.items():
        write_jsonl(split_records, output_dir / f"{split}.jsonl")

    # write_jsonl(records, output_dir / "all.jsonl")


def make_random_utterance_split(
    records: list[dict[str, Any]],
    train_ratio: float,
    dev_ratio: float,
    test_ratio: float,
    seed: int = 42,
    stratify_by_accent: bool = True,
) -> list[dict[str, Any]]:
    """
    Random utterance-level split.

    Speaker overlap is allowed across train/dev/test.
    This should mainly be used for debugging or as a reference setting.

    If stratify_by_accent=True, each accent is split separately so that
    train/dev/test have similar accent distributions.
    """
    validate_unsplit_metadata(records)

    if abs(train_ratio + dev_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + dev_ratio + test_ratio must be 1.0")

    rng = random.Random(seed)

    if stratify_by_accent:
        accent_to_records = defaultdict(list)

        for item in records:
            accent_to_records[item["accent_label"]].append(item)

        split_records = []

        for _, group_records in sorted(accent_to_records.items()):
            group_records = list(group_records)
            rng.shuffle(group_records)

            split_records.extend(
                _assign_random_split_to_group(
                    records=group_records,
                    train_ratio=train_ratio,
                    dev_ratio=dev_ratio,
                )
            )

    else:
        shuffled_records = list(records)
        rng.shuffle(shuffled_records)

        split_records = _assign_random_split_to_group(
            records=shuffled_records,
            train_ratio=train_ratio,
            dev_ratio=dev_ratio,
        )

    return sort_records(split_records)


def _assign_random_split_to_group(
    records: list[dict[str, Any]],
    train_ratio: float,
    dev_ratio: float,
) -> list[dict[str, Any]]:
    n = len(records)
    n_train = int(n * train_ratio)
    n_dev = int(n * dev_ratio)

    output = []

    for idx, item in enumerate(records):
        item = dict(item)

        if idx < n_train:
            item["split"] = "train"
        elif idx < n_train + n_dev:
            item["split"] = "dev"
        else:
            item["split"] = "test"

        output.append(item)

    return output


def make_speaker_independent_split(
    records: list[dict[str, Any]],
    train_ratio: float,
    dev_ratio: float,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Speaker-independent split.

    train/dev/test speakers are disjoint within each accent.
    """
    validate_unsplit_metadata(records)

    rng = random.Random(seed)

    accent_to_speakers = defaultdict(set)

    for item in records:
        accent_to_speakers[item["accent_label"]].add(item["speaker_id"])

    speaker_to_split = {}

    for accent, speakers in sorted(accent_to_speakers.items()):
        speakers = sorted(list(speakers))
        rng.shuffle(speakers)

        n = len(speakers)

        if n < 3:
            raise ValueError(
                f"Accent {accent} has only {n} speakers. "
                "Speaker-independent train/dev/test split requires at least 3 speakers."
            )

        n_train = max(1, int(n * train_ratio))
        n_dev = max(1, int(n * dev_ratio))

        if n_train + n_dev >= n:
            n_train = n - 2
            n_dev = 1

        train_speakers = speakers[:n_train]
        dev_speakers = speakers[n_train:n_train + n_dev]
        test_speakers = speakers[n_train + n_dev:]

        for speaker in train_speakers:
            speaker_to_split[speaker] = "train"
        for speaker in dev_speakers:
            speaker_to_split[speaker] = "dev"
        for speaker in test_speakers:
            speaker_to_split[speaker] = "test"

    output = []

    for item in records:
        item = dict(item)
        item["split"] = speaker_to_split[item["speaker_id"]]
        output.append(item)

    return sort_records(output)


def make_speaker_independent_test_utterance_dev_split(
    records: list[dict[str, Any]],
    test_speaker_ratio: float,
    dev_utterance_ratio: float,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Split strategy:

    1. Hold out test speakers within each accent.
    2. For the remaining speakers, randomly split utterances into train/dev.

    This means:
    - test speakers are unseen during training
    - train and dev may share speakers
    - dev is easier and more stable than fully speaker-independent dev
    """
    validate_unsplit_metadata(records)

    if not 0.0 < test_speaker_ratio < 1.0:
        raise ValueError("test_speaker_ratio must be between 0 and 1.")

    if not 0.0 < dev_utterance_ratio < 1.0:
        raise ValueError("dev_utterance_ratio must be between 0 and 1.")

    rng = random.Random(seed)

    accent_to_speakers = defaultdict(set)

    for item in records:
        accent_to_speakers[item["accent_label"]].add(item["speaker_id"])

    test_speakers = set()

    for accent, speakers in sorted(accent_to_speakers.items()):
        speakers = sorted(list(speakers))
        rng.shuffle(speakers)

        n = len(speakers)

        if n < 2:
            raise ValueError(
                f"Accent {accent} has only {n} speakers. "
                "Need at least 2 speakers to hold out test speakers."
            )

        n_test = max(1, int(round(n * test_speaker_ratio)))

        if n_test >= n:
            n_test = 1

        accent_test_speakers = speakers[:n_test]
        test_speakers.update(accent_test_speakers)

    test_records = []
    train_dev_pool = []

    for item in records:
        item = dict(item)

        if item["speaker_id"] in test_speakers:
            item["split"] = "test"
            test_records.append(item)
        else:
            train_dev_pool.append(item)

    accent_to_train_dev_records = defaultdict(list)

    for item in train_dev_pool:
        accent_to_train_dev_records[item["accent_label"]].append(item)

    train_records = []
    dev_records = []

    for _, group_records in sorted(accent_to_train_dev_records.items()):
        group_records = list(group_records)
        rng.shuffle(group_records)

        n = len(group_records)
        n_dev = max(1, int(round(n * dev_utterance_ratio)))

        dev_group = group_records[:n_dev]
        train_group = group_records[n_dev:]

        for item in dev_group:
            item = dict(item)
            item["split"] = "dev"
            dev_records.append(item)

        for item in train_group:
            item = dict(item)
            item["split"] = "train"
            train_records.append(item)

    output = train_records + dev_records + test_records
    return sort_records(output)


def sort_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        records,
        key=lambda x: (
            x.get("split", ""),
            x["accent_label"],
            x["speaker_id"],
            x["utt_id"],
        ),
    )


def summarize_split(records: list[dict[str, Any]]) -> dict[str, Any]:
    df = pd.DataFrame(records)

    if df.empty:
        raise ValueError("Cannot summarize an empty split.")

    if "split" not in df.columns:
        raise ValueError("Cannot summarize records without split field.")

    split_counts = df["split"].value_counts().to_dict()
    accent_counts = df["accent_label"].value_counts().to_dict()

    speaker_counts_by_split = (
        df.groupby("split")["speaker_id"].nunique().to_dict()
    )

    utterance_counts_by_accent_and_split = (
        pd.crosstab(df["accent_label"], df["split"]).to_dict()
    )

    speaker_counts_by_accent_and_split = (
        df.groupby(["accent_label", "split"])["speaker_id"]
        .nunique()
        .unstack(fill_value=0)
        .to_dict()
    )

    # Extra leakage diagnostics.
    train_speakers = set(df[df["split"] == "train"]["speaker_id"])
    dev_speakers = set(df[df["split"] == "dev"]["speaker_id"])
    test_speakers = set(df[df["split"] == "test"]["speaker_id"])

    summary = {
        "num_records": len(df),
        "split_counts": split_counts,
        "accent_counts": accent_counts,
        "speaker_counts_by_split": speaker_counts_by_split,
        "utterance_counts_by_accent_and_split": utterance_counts_by_accent_and_split,
        "speaker_counts_by_accent_and_split": speaker_counts_by_accent_and_split,
        "speaker_overlap": {
            "train_dev": sorted(train_speakers & dev_speakers),
            "train_test": sorted(train_speakers & test_speakers),
            "dev_test": sorted(dev_speakers & test_speakers),
        },
    }

    return summary


def save_summary(summary: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)