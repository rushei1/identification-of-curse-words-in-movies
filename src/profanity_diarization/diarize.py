"""Diarization stage (pyannote)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol

from .schemas import Turn


class Diarizer(Protocol):
    def diarize(self, wav: Path) -> list[Turn]: ...


class PyannoteDiarizer:
    def __init__(self, model: str = "pyannote/speaker-diarization", device: str = "auto"):
        self.model, self.device = model, device
        self._pipe = None

    def _load(self):
        if self._pipe is None:
            import torch
            from pyannote.audio import Pipeline

            token = os.environ.get("HF_TOKEN")  # None is fine when the model is cached
            try:
                pipe = Pipeline.from_pretrained(self.model, use_auth_token=token)
            except Exception as e:  # noqa: BLE001 - surface an actionable message
                raise RuntimeError(
                    f"cannot load {self.model!r}: set HF_TOKEN and accept the model terms on "
                    f"huggingface.co, or run `prefetch-models` once while online ({e})"
                ) from e
            if pipe is None:
                raise RuntimeError(f"{self.model!r} unavailable: check HF_TOKEN / accepted terms")
            dev = self.device
            if dev == "auto":
                dev = "cuda" if torch.cuda.is_available() else "cpu"
            pipe.to(torch.device(dev))
            self._pipe = pipe
        return self._pipe

    def diarize(self, wav: Path) -> list[Turn]:
        d = self._load()(str(wav))
        return [
            Turn(speaker=spk, start=t.start, end=t.end)
            for t, _, spk in d.itertracks(yield_label=True)
        ]
