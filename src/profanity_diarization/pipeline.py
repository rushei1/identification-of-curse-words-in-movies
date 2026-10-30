"""Orchestration with per-stage caching (key = input hash + stage config)."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from . import report as rp
from .asr import Transcriber, WhisperTranscriber, to_segments, transcribe_turns
from .audio import extract_audio
from .config import Settings
from .diarize import Diarizer, PyannoteDiarizer
from .gender import turn_gender
from .profanity import Detector, build_detector, counts
from .schemas import Report, Turn
from .speakers import SpeakerGrouper

log = logging.getLogger(__name__)


def file_hash(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _key(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]


def versions() -> dict:
    from importlib import metadata

    out = {}
    for pkg in ("openai-whisper", "pyannote.audio", "torch", "nltk"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = "not installed"
    return out


class Pipeline:
    def __init__(self, cfg: Settings, transcriber: Optional[Transcriber] = None,
                 diarizer: Optional[Diarizer] = None, detector: Optional[Detector] = None):  # fmt: skip
        self.cfg = cfg
        self.transcriber = transcriber or WhisperTranscriber(
            cfg.whisper_model, cfg.language, cfg.device
        )
        self.diarizer = diarizer or PyannoteDiarizer(cfg.diarization_model, cfg.device)
        self._detector = detector
        self.timings: dict = {}

    @property
    def detector(self) -> Detector:
        if self._detector is None:
            self._detector = build_detector(self.cfg)
        return self._detector

    def _stage(self, name, cache: Path, cfg_part, fn):
        """Run fn() unless cache/<key>_<name>.json exists. fn returns JSON-able data."""
        f = cache / f"{_key(name, cfg_part)}_{name}.json"
        t0 = time.perf_counter()
        if f.exists():
            log.info("%s: cached", name)
            data = json.loads(f.read_text(encoding="utf-8"))
        else:
            log.info("%s: running", name)
            data = fn()
            f.write_text(json.dumps(data), encoding="utf-8")
        self.timings[name] = round(time.perf_counter() - t0, 3)
        return data

    def process(self, src: Path, out_dir: Path) -> Report:
        src, out_dir = Path(src), Path(out_dir)
        cache = out_dir / ".cache"
        cache.mkdir(parents=True, exist_ok=True)
        self.timings = {}
        ih = file_hash(src)
        c = self.cfg

        t0 = time.perf_counter()
        wav = cache / f"{ih[:16]}.wav"
        if not wav.exists():
            extract_audio(src, wav)
        self.timings["audio"] = round(time.perf_counter() - t0, 3)

        turns = [
            Turn(**t)
            for t in self._stage(
                "diarize", cache, [ih, c.diarization_model],
                lambda: [t.model_dump() for t in self.diarizer.diarize(wav)],
            )
        ]  # fmt: skip
        texts = self._stage(
            "asr", cache, [ih, c.diarization_model, c.whisper_model, c.language],
            lambda: [x for _, x in transcribe_turns(self.transcriber, wav, turns, cache / "tmp")],
        )  # fmt: skip
        if c.speaker_labels == "pitch_gender":  # per TURN, from the audio (not per speaker label)
            tg = [turn_gender(wav, t.start, t.end) for t in turns]
            tgroups, f0s = [g for g, _ in tg], [f for _, f in tg]
        else:
            grouper = SpeakerGrouper(c.speaker_labels)
            tgroups, f0s = [grouper(t.speaker) for t in turns], None
        segs = to_segments(list(zip(turns, texts)), tgroups, f0s)
        hits = self.detector.detect(segs)
        groups = {
            "paper_gender": ["MALE", "FEMALE"],
            "pitch_gender": ["MALE", "FEMALE", "UNKNOWN"],
        }.get(c.speaker_labels) or sorted({s.group for s in segs})
        tot, uniq = counts(hits, groups)
        if not c.keep_intermediate:
            wav.unlink(missing_ok=True)
        return Report(
            source=str(src), preset=c.preset, created=datetime.now(timezone.utc).isoformat(),
            versions=versions(), config=c.model_dump(), segments=segs, hits=hits,
            counts=tot, unique_counts=uniq, timings=self.timings,
        )  # fmt: skip

    def write_outputs(self, r: Report, out_dir: Path, stem: str) -> list:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        d = self.detector
        paths = [out_dir / f"{stem}.json", out_dir / f"{stem}.csv",
                 out_dir / f"{stem}.srt", out_dir / f"{stem}.md"]  # fmt: skip
        rp.write_json(r, paths[0])
        rp.write_csv(r, paths[1])
        rp.write_srt(r, d, paths[2])
        rp.write_markdown(r, d, paths[3])
        return paths
