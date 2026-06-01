# src/data/text_normalizer.py

import re


def normalize_asr_text(text: str) -> str:
    """
    Basic ASR text normalization for WER/CER.

    This version is intentionally simple:
    - lowercase
    - remove punctuation
    - remove apostrophes
    - collapse spaces
    """
    if text is None:
        return ""

    text = str(text).lower().strip()

    # Keep letters, numbers, and spaces.
    text = re.sub(r"[^a-z0-9\s']", " ", text)

    # Normalize apostrophes:
    # don't -> dont
    text = text.replace("'", "")

    # Collapse multiple spaces.
    text = re.sub(r"\s+", " ", text).strip()

    return text