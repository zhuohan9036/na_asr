# src/evaluation/asr_metrics.py

from __future__ import annotations

from src.data.text_normalizer import normalize_asr_text


def _edit_distance(ref_tokens: list[str], hyp_tokens: list[str]) -> int:
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
                dp[i - 1][j] + 1,        # deletion
                dp[i][j - 1] + 1,        # insertion
                dp[i - 1][j - 1] + cost, # substitution
            )

    return dp[n][m]


def compute_wer(reference: str, prediction: str) -> float:
    reference = normalize_asr_text(reference)
    prediction = normalize_asr_text(prediction)

    ref_words = reference.split()
    hyp_words = prediction.split()

    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0

    return _edit_distance(ref_words, hyp_words) / len(ref_words)


def compute_cer(reference: str, prediction: str) -> float:
    reference = normalize_asr_text(reference).replace(" ", "")
    prediction = normalize_asr_text(prediction).replace(" ", "")

    ref_chars = list(reference)
    hyp_chars = list(prediction)

    if len(ref_chars) == 0:
        return 0.0 if len(hyp_chars) == 0 else 1.0

    return _edit_distance(ref_chars, hyp_chars) / len(ref_chars)