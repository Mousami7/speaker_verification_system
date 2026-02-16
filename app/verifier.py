"""Speaker verification using SpeechBrain ECAPA-TDNN embeddings."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import soundfile as sf
import torch
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier

TARGET_SR = 16000
FINE_TUNED_MODEL_PATH = Path(__file__).parent / "fine_tuned_speaker_verification_model_v1.pth"


def _prepare_audio(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Resample to 16kHz and convert to mono."""
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.dim() == 2 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != TARGET_SR:
        resampler = torchaudio.transforms.Resample(sample_rate, TARGET_SR)
        waveform = resampler(waveform)
    return waveform


def _load_audio_from_bytes(data: bytes) -> tuple[torch.Tensor, int]:
    """Load audio from bytes via soundfile (avoids torchaudio backend issues)."""
    audio, sample_rate = sf.read(io.BytesIO(data))
    waveform = torch.from_numpy(audio).float()
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    return waveform, sample_rate


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine similarity between two embedding vectors."""
    a_norm = a / (a.norm() + 1e-8)
    b_norm = b / (b.norm() + 1e-8)
    return float((a_norm * b_norm).sum())


class Verifier:
    """Loads ECAPA model once and provides verification."""

    def __init__(self, device: Optional[str] = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._classifier: Optional[EncoderClassifier] = None

    @property
    def classifier(self) -> EncoderClassifier:
        if self._classifier is None:
            raise RuntimeError("Model not loaded")
        return self._classifier

    def load_model(self) -> None:
        """Load SpeechBrain ECAPA model, then fine-tuned weights if present."""
        self._classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": self.device},
        )
        if FINE_TUNED_MODEL_PATH.exists():
            state_dict = torch.load(FINE_TUNED_MODEL_PATH, map_location=self.device, weights_only=True)
            self._classifier.load_state_dict(state_dict, strict=False)

    def encode(self, data: bytes) -> torch.Tensor:
        """Extract speaker embedding from audio bytes."""
        waveform, sample_rate = _load_audio_from_bytes(data)
        waveform = _prepare_audio(waveform, sample_rate)
        waveform = waveform.to(self.device)
        with torch.no_grad():
            emb = self.classifier.encode_batch(waveform)
        emb = emb.squeeze(0)
        return emb / (emb.norm() + 1e-8)

    def verify(
        self, data1: bytes, data2: bytes, threshold: float = 0.25
    ) -> tuple[float, bool]:
        """
        Compare two audio files. Returns (cosine_similarity, same_speaker).
        threshold: above this similarity = same speaker (tune via FAR/FRR/EER).
        """
        emb1 = self.encode(data1)
        emb2 = self.encode(data2)
        sim = cosine_similarity(emb1, emb2)
        return sim, sim >= threshold
