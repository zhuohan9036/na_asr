# src/models/wav2vec2_ctc_asr.py

from __future__ import annotations

from pathlib import Path

import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


def resolve_torch_dtype(torch_dtype: str | None) -> torch.dtype | None:
    """
    Convert config string to torch dtype.

    Supported:
    - None: use transformers default
    - "float32" / "fp32"
    - "float16" / "fp16" / "half"
    - "bfloat16" / "bf16"
    """
    if torch_dtype is None:
        return None

    torch_dtype = str(torch_dtype).lower()

    if torch_dtype in {"float32", "fp32"}:
        return torch.float32

    if torch_dtype in {"float16", "fp16", "half"}:
        return torch.float16

    if torch_dtype in {"bfloat16", "bf16"}:
        return torch.bfloat16

    raise ValueError(f"Unsupported torch_dtype: {torch_dtype}")


class Wav2Vec2CTCASR:
    """
    Wav2Vec2 CTC ASR wrapper for zero-shot inference.

    This works for checkpoints such as:
    - facebook/wav2vec2-base-960h
    - facebook/wav2vec2-large-960h-lv60-self
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        sampling_rate: int = 16000,
        torch_dtype: str | None = None,
    ):
        self.model_name = model_name
        self.sampling_rate = sampling_rate

        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"

        self.device = torch.device(device)

        resolved_dtype = resolve_torch_dtype(torch_dtype)

        self.processor = Wav2Vec2Processor.from_pretrained(model_name)

        if resolved_dtype is None:
            self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
        else:
            self.model = Wav2Vec2ForCTC.from_pretrained(
                model_name,
                torch_dtype=resolved_dtype,
            )

        self.model.to(self.device)
        self.model.eval()

        self.model_dtype = next(self.model.parameters()).dtype

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
    def transcribe_file(self, wav_path: str | Path) -> str:
        waveform = self.load_audio(wav_path)

        inputs = self.processor(
            waveform.cpu().numpy(),
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
            padding=True,
        )

        input_values = inputs.input_values.to(
            device=self.device,
            dtype=self.model_dtype,
        )

        attention_mask = None
        if hasattr(inputs, "attention_mask") and inputs.attention_mask is not None:
            attention_mask = inputs.attention_mask.to(self.device)

        outputs = self.model(
            input_values=input_values,
            attention_mask=attention_mask,
        )

        logits = outputs.logits
        predicted_ids = torch.argmax(logits, dim=-1)

        transcription = self.processor.batch_decode(predicted_ids)[0]

        return transcription.strip()