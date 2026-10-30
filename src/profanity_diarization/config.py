"""Settings = YAML preset + env + CLI overrides."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel

PRESET_DIR = Path(__file__).parent / "presets"


class Settings(BaseModel):
    preset: str = "paper"
    whisper_model: str = "small"
    language: Optional[str] = "en"
    diarization_model: str = "pyannote/speaker-diarization"
    device: Literal["auto", "cpu", "cuda"] = "auto"
    lexicon: Optional[str] = None  # path; None = bundled lexicon_b
    lemmatize: bool = True  # lemmatize(pos="v"), as final_gpu.py
    match: Literal["exact", "improved"] = "exact"  # exact = paper rule
    speaker_labels: Literal["paper_gender", "raw", "pitch_gender"] = "paper_gender"
    offline: bool = False
    keep_intermediate: bool = False


def load_settings(preset: str = "paper", **overrides) -> Settings:
    path = Path(preset) if preset.endswith((".yaml", ".yml")) else PRESET_DIR / f"{preset}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"unknown preset {preset!r} ({path})")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    data.update({k: v for k, v in overrides.items() if v is not None})
    data.setdefault("preset", path.stem)
    return Settings(**data)
