# src/data/sandi.py

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


SANDI_SPLITS = ("train", "dev", "eval")


def read_flist_tsv(path: str | Path) -> dict[str, str]:
    """
    Read SANDI flist tsv.

    Format:
        utt_id    relative_audio_path
    """
    path = Path(path)
    utt_to_relpath: dict[str, str] = {}

    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"Bad flist line at {path}:{line_idx}: {line}")

            utt_id, rel_path = parts
            if utt_id in utt_to_relpath:
                raise ValueError(f"Duplicate utt_id in flist {path}: {utt_id}")

            utt_to_relpath[utt_id] = rel_path

    return utt_to_relpath


def read_trans_ref_json(path: str | Path) -> dict[str, dict[str, Any]]:
    """
    Read SANDI transcription reference JSON.

    Expected structure:
        {
          "ref-type": "transcription",
          "files": [
            {
              "File-id": "...",
              "Transcript": [...],
              "Question": "..."
            }
          ]
        }
    """
    path = Path(path)

    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    files = obj.get("files")
    if not isinstance(files, list):
        raise ValueError(f"Expected key 'files' to be a list in {path}")

    utt_to_item: dict[str, dict[str, Any]] = {}

    for item in files:
        utt_id = item.get("File-id")
        if not utt_id:
            raise ValueError(f"Missing File-id in {path}: {item}")

        if utt_id in utt_to_item:
            raise ValueError(f"Duplicate File-id in {path}: {utt_id}")

        utt_to_item[utt_id] = item

    return utt_to_item


def parse_question(question_raw: Any) -> tuple[str | None, int | None]:
    """
    Parse the Question field.

    It is often a JSON string like:
        {"text": "...", "speaking-time": 20}
    """
    if question_raw is None:
        return None, None

    if isinstance(question_raw, dict):
        question_obj = question_raw
    elif isinstance(question_raw, str):
        try:
            question_obj = json.loads(question_raw)
        except json.JSONDecodeError:
            return question_raw, None
    else:
        return str(question_raw), None

    question_text = question_obj.get("text")
    speaking_time = question_obj.get("speaking-time")

    return question_text, speaking_time


def parse_transcript_items(
    transcript_items: list[dict[str, Any]],
    keep_partial: bool,
) -> tuple[str, int, dict[str, int], int]:
    """
    Convert SANDI token-list transcript into plain ASR reference text.

    Rules:
    - keep all item["word"] tokens
    - optionally drop words marked as partial
    - ignore item["tag"] tokens for ASR reference
    - count partial words and tags for later analysis
    """
    words: list[str] = []
    partial_word_count = 0
    tag_counter: Counter[str] = Counter()

    for item in transcript_items:
        if "word" in item:
            word = str(item["word"]).strip()
            marks = item.get("marks", [])

            is_partial = isinstance(marks, list) and "partial" in marks

            if is_partial:
                partial_word_count += 1
                if not keep_partial:
                    continue

            if word:
                words.append(word)

        elif "tag" in item:
            tag = str(item["tag"]).strip()
            if tag:
                tag_counter[tag] += 1

    transcript = " ".join(words)
    return transcript, partial_word_count, dict(tag_counter), len(words)


def infer_speaker_and_prompt_id(utt_id: str) -> tuple[str, str]:
    """
    Infer speaker/session id and prompt id from utterance id.

    Example:
        SI114J-00026-P10005
        speaker_id = SI114J-00026
        prompt_id  = P10005
    """
    if "-" not in utt_id:
        return utt_id, "unknown"

    speaker_id, prompt_id = utt_id.rsplit("-", 1)
    return speaker_id, prompt_id


def resolve_audio_path(sandi_root: Path, relative_audio_path: str) -> tuple[Path, bool]:
    """
    Resolve audio path robustly.

    Most train/dev paths are:
        sandi_root/data/flac/...

    Eval may be unpacked as:
        sandi_root/eval-data-release-20250327/sandi2025-challenge/data/flac/eval/...
    """
    candidates = [
        sandi_root / relative_audio_path,
        sandi_root / "eval-data-release-20250327" / "sandi2025-challenge" / relative_audio_path,
        sandi_root / "sandi2025-challenge" / relative_audio_path,
    ]

    for path in candidates:
        if path.exists():
            return path.resolve(), True

    return candidates[0].resolve(), False


def build_sandi_metadata_for_split(
    sandi_root: str | Path,
    split: str,
    dataset_name: str = "SpeakAndImprove2025",
    strict: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Build metadata records for one official split: train/dev/eval.
    """
    if split not in SANDI_SPLITS:
        raise ValueError(f"Unknown SANDI split: {split}")

    sandi_root = Path(sandi_root).resolve()

    flist_path = sandi_root / "reference-materials" / "flists.flac" / f"{split}-asr.tsv"
    trans_path = sandi_root / "reference-materials" / "annotations" / f"{split}-trans-ref.json"

    if not flist_path.exists():
        raise FileNotFoundError(f"Missing flist file: {flist_path}")

    if not trans_path.exists():
        raise FileNotFoundError(f"Missing transcription reference: {trans_path}")

    utt_to_relpath = read_flist_tsv(flist_path)
    utt_to_trans_item = read_trans_ref_json(trans_path)

    records: list[dict[str, Any]] = []

    missing_transcript: list[str] = []
    missing_audio: list[str] = []
    empty_transcript: list[str] = []

    for utt_id, rel_audio_path in sorted(utt_to_relpath.items()):
        trans_item = utt_to_trans_item.get(utt_id)

        if trans_item is None:
            missing_transcript.append(utt_id)
            if strict:
                raise KeyError(f"Missing transcript for utt_id={utt_id}")
            continue

        audio_path, audio_exists = resolve_audio_path(sandi_root, rel_audio_path)
        if not audio_exists:
            missing_audio.append(utt_id)
            if strict:
                raise FileNotFoundError(f"Missing audio for utt_id={utt_id}: {audio_path}")
            continue

        transcript_items = trans_item.get("Transcript", [])
        if not isinstance(transcript_items, list):
            raise ValueError(f"Transcript must be a list for utt_id={utt_id}")

        transcript_keep_partial, partial_count, tag_counts, word_count_keep = parse_transcript_items(
            transcript_items=transcript_items,
            keep_partial=True,
        )
        transcript_drop_partial, _, _, word_count_drop = parse_transcript_items(
            transcript_items=transcript_items,
            keep_partial=False,
        )

        if not transcript_keep_partial:
            empty_transcript.append(utt_id)
            if strict:
                raise ValueError(f"Empty transcript for utt_id={utt_id}")
            continue

        speaker_id, prompt_id = infer_speaker_and_prompt_id(utt_id)

        question_text, speaking_time = parse_question(trans_item.get("Question"))

        record = {
            "utt_id": utt_id,
            "speaker_id": speaker_id,
            "prompt_id": prompt_id,
            "dataset_name": dataset_name,
            "official_split": split,
            "split": split,
            "wav_path": str(audio_path),
            "relative_audio_path": rel_audio_path,

            # Default ASR reference.
            "transcript": transcript_keep_partial,

            # Alternative references.
            "transcript_word_keep_partial": transcript_keep_partial,
            "transcript_word_drop_partial": transcript_drop_partial,

            # Useful analysis metadata.
            "partial_word_count": partial_count,
            "word_count_keep_partial": word_count_keep,
            "word_count_drop_partial": word_count_drop,
            "tag_counts": tag_counts,
            "hesitation_count": tag_counts.get("hesitation", 0),

            "question_text": question_text,
            "speaking_time": speaking_time,
        }

        records.append(record)

    transcript_without_audio = sorted(set(utt_to_trans_item) - set(utt_to_relpath))

    summary = {
        "split": split,
        "flist_path": str(flist_path),
        "transcript_path": str(trans_path),
        "audio_entries": len(utt_to_relpath),
        "transcript_entries": len(utt_to_trans_item),
        "matched_records": len(records),
        "missing_transcript_count": len(missing_transcript),
        "missing_audio_count": len(missing_audio),
        "empty_transcript_count": len(empty_transcript),
        "transcript_without_audio_count": len(transcript_without_audio),
        "missing_transcript_examples": missing_transcript[:10],
        "missing_audio_examples": missing_audio[:10],
        "empty_transcript_examples": empty_transcript[:10],
        "transcript_without_audio_examples": transcript_without_audio[:10],
    }

    return records, summary