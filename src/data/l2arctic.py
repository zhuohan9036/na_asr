# src/data/l2arctic.py

from pathlib import Path
import wave


# 你可以先只写 prototype speaker mapping。
# 后面再慢慢补全。
L2ARCTIC_SPEAKER_TO_ACCENT = {
    # Arabic
    "ABA": "Arabic",
    "SKA": "Arabic",
    "YBAA": "Arabic",
    "ZHAA": "Arabic",

    # Mandarin
    "BWC": "Mandarin",
    "LXC": "Mandarin",
    "NCC": "Mandarin",
    "TXHC": "Mandarin",

    # Hindi / Indian English
    "ASI": "Hindi",
    "RRBI": "Hindi",
    "SVBI": "Hindi",
    "TNI": "Hindi",

    # Korean
    "HJK": "Korean",
    "HKK": "Korean",
    "YDCK": "Korean",
    "YKWK": "Korean",

    # Spanish
    "EBVS": "Spanish",
    "ERMS": "Spanish",
    "MBMPS": "Spanish",
    "NJS": "Spanish",
}


def get_wav_duration_seconds(wav_path: str | Path) -> float:
    """
    Get wav duration using the standard library.
    This avoids depending on torchaudio/librosa at the metadata stage.
    """
    wav_path = Path(wav_path)

    with wave.open(str(wav_path), "rb") as wf:
        frames = wf.getnframes()
        sample_rate = wf.getframerate()

    return frames / float(sample_rate)


def find_transcript_file(speaker_dir: Path, utt_id: str) -> Path | None:
    """
    Try common transcript locations in L2-ARCTIC.
    """
    candidate_paths = [
        speaker_dir / "transcript" / f"{utt_id}.txt",
        speaker_dir / "transcripts" / f"{utt_id}.txt",
        speaker_dir / "txt" / f"{utt_id}.txt",
    ]

    for path in candidate_paths:
        if path.exists():
            return path

    return None


def read_transcript(path: Path) -> str:
    text = path.read_text(encoding="utf-8").strip()
    return " ".join(text.split())


def build_l2arctic_metadata(
    root_dir: str | Path,
    keep_speakers: list[str] | None = None,
) -> list[dict]:
    """
    Build metadata records from L2-ARCTIC.

    Each record has:
    - utt_id
    - speaker_id
    - accent_label
    - native_language
    - dataset_name
    - wav_path
    - transcript
    - duration
    """
    root_dir = Path(root_dir).expanduser().resolve()

    if not root_dir.exists():
        raise FileNotFoundError(f"L2-ARCTIC root not found: {root_dir}")

    records = []

    speaker_dirs = [p for p in root_dir.iterdir() if p.is_dir()]
    speaker_dirs = sorted(speaker_dirs, key=lambda p: p.name)

    if keep_speakers is not None:
        keep_speakers = set(keep_speakers)
        speaker_dirs = [p for p in speaker_dirs if p.name in keep_speakers]

    for speaker_dir in speaker_dirs:
        speaker_id = speaker_dir.name

        if speaker_id not in L2ARCTIC_SPEAKER_TO_ACCENT:
            continue

        accent_label = L2ARCTIC_SPEAKER_TO_ACCENT[speaker_id]

        wav_dir = speaker_dir / "wav"
        if not wav_dir.exists():
            continue

        wav_paths = sorted(wav_dir.glob("*.wav"))

        for wav_path in wav_paths:
            utt_id = wav_path.stem

            transcript_path = find_transcript_file(speaker_dir, utt_id)
            if transcript_path is None:
                continue

            transcript = read_transcript(transcript_path)

            try:
                duration = get_wav_duration_seconds(wav_path)
            except Exception:
                duration = None

            record = {
                "utt_id": utt_id,
                "speaker_id": speaker_id,
                "accent_label": accent_label,
                "native_language": accent_label,
                "dataset_name": "L2-ARCTIC",
                "wav_path": str(wav_path),
                "transcript": transcript,
                "duration": duration,
            }

            records.append(record)

    return records