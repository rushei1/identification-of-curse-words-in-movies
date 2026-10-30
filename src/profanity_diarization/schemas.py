"""Pydantic models for every stage boundary (each stage is testable/cacheable alone)."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class Turn(BaseModel):
    """One diarization turn (pyannote ``itertracks`` row)."""

    speaker: str
    start: float
    end: float


class Segment(BaseModel):
    """Transcribed text of one speaker turn (legacy transcribes per turn)."""

    speaker: str
    group: str  # legacy "gender" label (heuristic) or the raw speaker label
    start: float
    end: float
    text: str
    f0: Optional[float] = None  # median pitch (Hz) when speaker_labels=pitch_gender


class ProfanityHit(BaseModel):
    term: str  # lexicon entry matched (raw; mask before putting in docs)
    token: str  # token as spoken, lowercased
    speaker: str
    group: str
    start: float  # turn start: legacy keeps no word-level timestamps
    end: float
    context: str


class Report(BaseModel):
    source: str
    preset: str
    created: str
    versions: dict[str, str] = Field(default_factory=dict)
    config: dict = Field(default_factory=dict)
    segments: list[Segment]
    hits: list[ProfanityHit]
    counts: dict[str, int]  # per group
    unique_counts: dict[str, int]
    timings: Optional[dict[str, float]] = None
