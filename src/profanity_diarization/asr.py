"""ASR stage. Legacy transcribes each diarization turn separately (no word timestamps)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Protocol

from .audio import slice_wav
from .schemas import Segment, Turn


class Transcriber(Protocol):
    def transcribe_file(self, wav: Path) -> str: ...


class WhisperTranscriber:
    def __init__(self, model: str = "small", language: Optional[str] = "en", device: str = "auto"):
        self.model_name, self.language, self.device = model, language, device
        self._model = None

    def _load(self):
        if self._model is None:  # lazy: heavy import
            import torch
            import whisper

            dev = self.device
            if dev == "auto":
                dev = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = whisper.load_model(self.model_name, device=dev)
        return self._model

    def transcribe_file(self, wav: Path) -> str:
        return self._load().transcribe(str(wav), language=self.language)["text"].strip()


def transcribe_turns(
    transcriber: Transcriber, wav: Path, turns: list[Turn], tmp: Path
) -> list[tuple[Turn, str]]:
    tmp.mkdir(parents=True, exist_ok=True)
    out = []
    for i, t in enumerate(turns):
        piece = tmp / f"turn_{i}.wav"
        try:
            slice_wav(wav, t.start, t.end, piece)
            out.append((t, transcriber.transcribe_file(piece)))
        finally:
            piece.unlink(missing_ok=True)
    return out


def to_segments(rows: list[tuple[Turn, str]], groups: list, f0s: Optional[list] = None) -> list[Segment]:
    """Keep only non-empty text (legacy ``if text:``). ``groups[i]`` belongs to ``rows[i]``."""
    f0s = f0s or [None] * len(rows)
    return [
        Segment(speaker=t.speaker, group=g, start=t.start, end=t.end, text=x, f0=f)
        for (t, x), g, f in zip(rows, groups, f0s)
        if x
    ]
