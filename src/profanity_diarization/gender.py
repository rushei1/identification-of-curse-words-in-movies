"""Per-turn speaker gender from voice pitch (median F0). Opt-in: ``speaker_labels: pitch_gender``.

A rough acoustic heuristic, NOT a trained classifier: F0 overlaps between low female and high
male voices, children read as FEMALE, music/noise can bias it. Each turn also keeps its F0 so
the result can be audited. Turns with too little voiced audio are ``UNKNOWN``.
"""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Optional

import numpy as np

THRESHOLD_HZ = 165.0  # common male/female midpoint
MIN_VOICED_FRAMES = 20  # 0.2 s of voiced speech


def median_f0(x: np.ndarray, sr: int) -> tuple:
    """Median autocorrelation F0 (70-400 Hz) over voiced frames -> (f0 or None, n_voiced)."""
    frame, hop = int(0.04 * sr), int(0.01 * sr)
    lo, hi = int(sr / 400), int(sr / 70)
    if len(x) < frame:
        return None, 0
    starts = range(0, len(x) - frame + 1, hop)
    rms = np.array([np.sqrt(np.mean(x[i : i + frame] ** 2)) for i in starts])
    gate = max(0.15 * np.percentile(rms, 95), 1e-4)
    f0s = []
    for i, e in zip(starts, rms):
        if e < gate:
            continue
        w = x[i : i + frame] - x[i : i + frame].mean()
        ac = np.fft.irfft(np.abs(np.fft.rfft(w, 2 * frame)) ** 2)[:frame]
        if ac[0] <= 0:
            continue
        seg = ac[lo : hi + 1]
        if seg.max() / ac[0] < 0.4:  # unvoiced
            continue
        k = int(np.argmax(seg >= 0.85 * seg.max()))  # first strong peak: avoids octave-down errors
        f0s.append(sr / (lo + k))
    return (float(np.median(f0s)) if f0s else None), len(f0s)


def classify(f0: Optional[float], n_voiced: int) -> str:
    if f0 is None or n_voiced < MIN_VOICED_FRAMES:
        return "UNKNOWN"
    return "MALE" if f0 < THRESHOLD_HZ else "FEMALE"


def read_slice(src: Path, start_s: float, end_s: float) -> tuple:
    with wave.open(str(src), "rb") as w:
        sr, n = w.getframerate(), w.getnframes()
        a, b = min(int(start_s * sr), n), min(int(end_s * sr), n)
        w.setpos(a)
        raw = w.readframes(max(b - a, 0))
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0, sr


def turn_gender(wav: Path, start: float, end: float) -> tuple:
    """-> (label, f0_hz or None) for one turn of a 16-bit mono wav."""
    x, sr = read_slice(wav, start, end)
    f0, n = median_f0(x, sr)
    return classify(f0, n), (round(f0, 1) if f0 else None)


def smooth_by_speaker(groups: list, speakers: list, min_frac: float = 0.8) -> list:
    """If >= ``min_frac`` of a diarization speaker's decided turns agree, give them all that label.

    Reduces per-turn pitch noise when the diarizer is clean; leaves mixed speakers per-turn.
    """
    out = list(groups)
    for spk in set(speakers):
        idx = [i for i, s in enumerate(speakers) if s == spk]
        known = [groups[i] for i in idx if groups[i] in ("MALE", "FEMALE")]
        for lab in ("MALE", "FEMALE"):
            if known and known.count(lab) / len(known) >= min_frac:
                for i in idx:
                    out[i] = lab
    return out
