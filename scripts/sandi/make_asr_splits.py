# scripts/sandi/make_asr_splits.py

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.utils.config import load_config
from src.utils.logging import setup_logger


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return records


def write_jsonl(records: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(obj: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def copy_with_split(records: list[dict[str, Any]], split_name: str) -> list[dict[str, Any]]:
    new_records = []
    for record in records:
        record = dict(record)
        record["split"] = split_name
        new_records.append(record)
    return new_records


def assign_random_utterance_split(
    records: list[dict[str, Any]],
    train_ratio: float,
    dev_ratio: float,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(seed)

    records = [dict(r) for r in records]
    rng.shuffle(records)

    n = len(records)
    n_train = int(n * train_ratio)
    n_dev = int(n * dev_ratio)

    train = copy_with_split(records[:n_train], "train")
    dev = copy_with_split(records[n_train:n_train + n_dev], "dev")
    test = copy_with_split(records[n_train + n_dev:], "test")

    return {"train": train, "dev": dev, "test": test}


def group_by_speaker(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    speaker_to_records: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for record in records:
        speaker_to_records[record["speaker_id"]].append(record)

    return dict(speaker_to_records)


def split_speakers(
    speakers: list[str],
    train_ratio: float,
    dev_ratio: float,
    seed: int,
) -> tuple[set[str], set[str], set[str]]:
    rng = random.Random(seed)

    speakers = list(speakers)
    rng.shuffle(speakers)

    n = len(speakers)
    n_train = int(n * train_ratio)
    n_dev = int(n * dev_ratio)

    # Keep at least one dev/test speaker if possible.
    if n >= 3:
        n_dev = max(1, n_dev)
        n_test = max(1, n - n_train - n_dev)

        while n_train + n_dev + n_test > n:
            n_train -= 1
    else:
        n_test = max(1, n - n_train - n_dev)

    train_speakers = set(speakers[:n_train])
    dev_speakers = set(speakers[n_train:n_train + n_dev])
    test_speakers = set(speakers[n_train + n_dev:])

    return train_speakers, dev_speakers, test_speakers


def assign_speaker_independent_split(
    records: list[dict[str, Any]],
    train_ratio: float,
    dev_ratio: float,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    speaker_to_records = group_by_speaker(records)

    train_speakers, dev_speakers, test_speakers = split_speakers(
        speakers=sorted(speaker_to_records),
        train_ratio=train_ratio,
        dev_ratio=dev_ratio,
        seed=seed,
    )

    split_records = {"train": [], "dev": [], "test": []}

    for speaker, speaker_records in speaker_to_records.items():
        if speaker in train_speakers:
            split_records["train"].extend(copy_with_split(speaker_records, "train"))
        elif speaker in dev_speakers:
            split_records["dev"].extend(copy_with_split(speaker_records, "dev"))
        elif speaker in test_speakers:
            split_records["test"].extend(copy_with_split(speaker_records, "test"))
        else:
            raise RuntimeError(f"Speaker was not assigned to a split: {speaker}")

    return split_records


def assign_speaker_independent_test_utterance_dev_split(
    records: list[dict[str, Any]],
    test_speaker_ratio: float,
    dev_ratio: float,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    """
    Hold out test speakers.
    Then sample dev at utterance level from all non-test speakers.
    """
    rng = random.Random(seed)

    speaker_to_records = group_by_speaker(records)
    speakers = sorted(speaker_to_records)
    rng.shuffle(speakers)

    n_speakers = len(speakers)
    n_test = max(1, int(n_speakers * test_speaker_ratio))

    test_speakers = set(speakers[:n_test])
    train_dev_speakers = set(speakers[n_test:])

    test_records = []
    train_dev_records = []

    for speaker, speaker_records in speaker_to_records.items():
        if speaker in test_speakers:
            test_records.extend(speaker_records)
        elif speaker in train_dev_speakers:
            train_dev_records.extend(speaker_records)
        else:
            raise RuntimeError(f"Speaker was not assigned: {speaker}")

    rng.shuffle(train_dev_records)

    n_dev = int(len(train_dev_records) * dev_ratio)

    dev_records = train_dev_records[:n_dev]
    train_records = train_dev_records[n_dev:]

    return {
        "train": copy_with_split(train_records, "train"),
        "dev": copy_with_split(dev_records, "dev"),
        "test": copy_with_split(test_records, "test"),
    }


def summarize_split(split_records: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}

    for split, records in split_records.items():
        speakers = {r["speaker_id"] for r in records}
        prompts = {r["prompt_id"] for r in records}

        summary[split] = {
            "utterance_count": len(records),
            "speaker_count": len(speakers),
            "prompt_count": len(prompts),
        }

    try:
        import pandas as pd

        all_records = []
        for split, records in split_records.items():
            for record in records:
                r = dict(record)
                r["_assigned_split"] = split
                all_records.append(r)

        df = pd.DataFrame(all_records)

        summary["split_x_official_split"] = pd.crosstab(
            df["_assigned_split"],
            df["official_split"],
        ).to_dict()

        summary["split_x_prompt_id"] = pd.crosstab(
            df["_assigned_split"],
            df["prompt_id"],
        ).to_dict()

    except Exception as e:
        summary["pandas_summary_error"] = str(e)

    return summary


def save_split(
    split_records: dict[str, list[dict[str, Any]]],
    output_dir: Path,
    summary_path: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for split, records in split_records.items():
        write_jsonl(records, output_dir / f"{split}.jsonl")

    summary = summarize_split(split_records)
    write_json(summary, summary_path)


def main() -> None:
    config = load_config()

    experiment_name = config["experiment"]["name"]
    logger = setup_logger(experiment_name)

    metadata_path = Path(config["data"]["metadata_path"])
    output_root = Path(config["output"]["manifest_root"])
    table_dir = Path(config["output"]["table_dir"])

    seed = int(config.get("seed", 42))
    train_ratio = float(config["split"].get("train_ratio", 0.8))
    dev_ratio = float(config["split"].get("dev_ratio", 0.1))
    test_speaker_ratio = float(config["split"].get("test_speaker_ratio", 0.1))

    logger.info("Making SANDI ASR splits.")
    logger.info(f"Config path: {config['_config_path']}")
    logger.info(f"Metadata path: {metadata_path}")
    logger.info(f"Output root: {output_root}")

    records = read_jsonl(metadata_path)
    logger.info(f"Loaded records: {len(records)}")

    random_utterance = assign_random_utterance_split(
        records=records,
        train_ratio=train_ratio,
        dev_ratio=dev_ratio,
        seed=seed,
    )
    save_split(
        random_utterance,
        output_dir=output_root / "random_utterance",
        summary_path=table_dir / "random_utterance_split_summary.json",
    )
    logger.info("Saved random_utterance split.")

    speaker_independent = assign_speaker_independent_split(
        records=records,
        train_ratio=train_ratio,
        dev_ratio=dev_ratio,
        seed=seed,
    )
    save_split(
        speaker_independent,
        output_dir=output_root / "speaker_independent",
        summary_path=table_dir / "speaker_independent_split_summary.json",
    )
    logger.info("Saved speaker_independent split.")

    si_test_utt_dev = assign_speaker_independent_test_utterance_dev_split(
        records=records,
        test_speaker_ratio=test_speaker_ratio,
        dev_ratio=dev_ratio,
        seed=seed,
    )
    save_split(
        si_test_utt_dev,
        output_dir=output_root / "speaker_independent_test_utterance_dev",
        summary_path=table_dir / "speaker_independent_test_utterance_dev_split_summary.json",
    )
    logger.info("Saved speaker_independent_test_utterance_dev split.")

    logger.info("Finished making SANDI splits.")


if __name__ == "__main__":
    main()