# src/evaluation/asr_metrics.py

from __future__ import annotations

from dataclasses import dataclass

from src.data.text_normalizer import normalize_asr_text


@dataclass
class ErrorRateStats:
    """
    Store edit-distance statistics for one reference-prediction pair.

    errors:
        edit distance between reference and prediction.
        For WER, this is word-level edit distance.
        For CER, this is character-level edit distance.

    reference_length:
        number of reference units.
        For WER, this is number of reference words.
        For CER, this is number of reference characters.
    """

    errors: int
    reference_length: int

    @property
    def error_rate(self) -> float:
        if self.reference_length == 0:
            return 0.0 if self.errors == 0 else 1.0

        return self.errors / self.reference_length


def _edit_distance(ref_tokens: list[str], hyp_tokens: list[str]) -> int:
    """
    Compute Levenshtein edit distance.

    Operations:
    - substitution
    - deletion
    - insertion
    """
    n = len(ref_tokens)
    m = len(hyp_tokens)

    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i

    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                cost = 0
            else:
                cost = 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,         # deletion
                dp[i][j - 1] + 1,         # insertion
                dp[i - 1][j - 1] + cost,  # substitution
            )

    return dp[n][m]


def compute_wer_stats(reference: str, prediction: str) -> ErrorRateStats:
    """
    Compute word-level edit statistics for one utterance.
    """
    reference = normalize_asr_text(reference)
    prediction = normalize_asr_text(prediction)

    ref_words = reference.split()
    hyp_words = prediction.split()

    errors = _edit_distance(ref_words, hyp_words)

    return ErrorRateStats(
        errors=errors,
        reference_length=len(ref_words),
    )


def compute_cer_stats(reference: str, prediction: str) -> ErrorRateStats:
    """
    Compute character-level edit statistics for one utterance.

    Spaces are removed before CER computation.
    """
    reference = normalize_asr_text(reference).replace(" ", "")
    prediction = normalize_asr_text(prediction).replace(" ", "")

    ref_chars = list(reference)
    hyp_chars = list(prediction)

    errors = _edit_distance(ref_chars, hyp_chars)

    return ErrorRateStats(
        errors=errors,
        reference_length=len(ref_chars),
    )


def compute_wer(reference: str, prediction: str) -> float:
    """
    Utterance-level WER.
    """
    return compute_wer_stats(reference, prediction).error_rate


def compute_cer(reference: str, prediction: str) -> float:
    """
    Utterance-level CER.
    """
    return compute_cer_stats(reference, prediction).error_rate


def compute_corpus_error_rate(stats_list: list[ErrorRateStats]) -> float:
    """
    Compute corpus-level error rate.

    Formula:
        sum(errors) / sum(reference_length)

    This is the standard way to aggregate WER/CER over a dataset.
    """
    total_errors = sum(item.errors for item in stats_list)
    total_reference_length = sum(item.reference_length for item in stats_list)

    if total_reference_length == 0:
        return 0.0 if total_errors == 0 else 1.0

    return total_errors / total_reference_length