# src/models/whisper_asr.py

from __future__ import annotations

from pathlib import Path

import torch
import torchaudio
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def resolve_torch_dtype(torch_dtype: str | None) -> torch.dtype | str | None:
    """
    Convert config string to torch dtype.

    Supported:
    - null / None: use transformers default
    - "auto": let transformers decide
    - "float32"
    - "float16"
    - "bfloat16"
    """
    if torch_dtype is None:
        return None

    torch_dtype = str(torch_dtype).lower()

    if torch_dtype == "auto":
        return "auto"

    if torch_dtype in {"float32", "fp32"}:
        return torch.float32

    if torch_dtype in {"float16", "fp16", "half"}:
        return torch.float16

    if torch_dtype in {"bfloat16", "bf16"}:
        return torch.bfloat16

    raise ValueError(f"Unsupported torch_dtype: {torch_dtype}")


class WhisperASR:
    """
    Simple Whisper ASR wrapper for zero-shot inference.

    Compatible with Whisper-small, Whisper-medium, and Whisper-large-v3.

    The key compatibility point:
    input_features are cast to the same dtype as the model parameters.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        language: str = "en",
        task: str = "transcribe",
        sampling_rate: int = 16000,
        torch_dtype: str | None = None,
    ):
        self.model_name = model_name
        self.language = language
        self.task = task
        self.sampling_rate = sampling_rate

        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"

        self.device = torch.device(device)

        resolved_dtype = resolve_torch_dtype(torch_dtype)

        self.processor = WhisperProcessor.from_pretrained(model_name)

        if resolved_dtype is None:
            self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        else:
            self.model = WhisperForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=resolved_dtype,
            )

        self.model.to(self.device)
        self.model.eval()

        # Actual dtype after loading and moving model.
        self.model_dtype = next(self.model.parameters()).dtype

        self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=language,
            task=task,
        )

    def load_audio(self, wav_path: str | Path) -> torch.Tensor:
        wav_path = Path(wav_path)

        if not wav_path.exists():
            raise FileNotFoundError(f"Wav file not found: {wav_path}")

        waveform, sample_rate = torchaudio.load(str(wav_path))

        # Convert multi-channel audio to mono.
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        waveform = waveform.squeeze(0)

        if sample_rate != self.sampling_rate:
            waveform = torchaudio.functional.resample(
                waveform,
                orig_freq=sample_rate,
                new_freq=self.sampling_rate,
            )

        return waveform

    @torch.no_grad()
    def transcribe_file(
        self,
        wav_path: str | Path,
        max_new_tokens: int = 128,
    ) -> str:
        waveform = self.load_audio(wav_path)

        # WhisperProcessor expects CPU numpy float waveform.
        inputs = self.processor(
            waveform.cpu().numpy(),
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
        )

        # Important:
        # Cast input_features to the same dtype as the loaded model.
        # This avoids errors like:
        # "Input type (float) and bias type (c10::Half) should be the same"
        input_features = inputs.input_features.to(
            device=self.device,
            dtype=self.model_dtype,
        )

        predicted_ids = self.model.generate(
            input_features,
            forced_decoder_ids=self.forced_decoder_ids,
            max_new_tokens=max_new_tokens,
        )

        transcription = self.processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True,
        )[0]

        return transcription.strip()