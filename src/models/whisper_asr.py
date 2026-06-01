# src/models/whisper_asr.py

from __future__ import annotations

from pathlib import Path

import torch
import torchaudio
from transformers import WhisperForConditionalGeneration, WhisperProcessor


class WhisperASR:
    """
    Simple Whisper ASR wrapper for zero-shot inference.

    Input:
        wav_path

    Output:
        predicted transcript
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        language: str = "en",
        task: str = "transcribe",
        sampling_rate: int = 16000,
    ):
        self.model_name = model_name
        self.language = language
        self.task = task
        self.sampling_rate = sampling_rate

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

        self.device = torch.device(device)

        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

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

        inputs = self.processor(
            waveform.numpy(),
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
        )

        input_features = inputs.input_features.to(self.device)

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