"""ffmpeg wrapper (no shell strings) and wav slicing."""

from __future__ import annotations

import shutil
import subprocess
import wave
from pathlib import Path


class AudioError(RuntimeError):
    pass


def ffmpeg_cmd(src: Path, dst: Path, ffmpeg: str = "ffmpeg") -> list[str]:
    """16 kHz mono 16-bit PCM wav, video stream dropped (as the legacy pydub/ffmpeg export)."""
    return [ffmpeg, "-y", "-i", str(src), "-vn", "-acodec", "pcm_s16le", "-ar", "16000",
            "-ac", "1", str(dst)]  # fmt: skip


def extract_audio(src: Path, dst: Path) -> Path:
    src, dst = Path(src), Path(dst)
    if not src.is_file():
        raise AudioError(f"input not found: {src}")
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise AudioError("ffmpeg not found on PATH; install it from https://ffmpeg.org")
    dst.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(ffmpeg_cmd(src, dst, exe), capture_output=True, text=True)
    if r.returncode != 0:
        raise AudioError(f"ffmpeg failed on {src}:\n{r.stderr[-500:]}")
    return dst


def slice_wav(src: Path, start_s: float, end_s: float, dst: Path) -> Path:
    """Cut like legacy ``audio[int(s*1000):int(e*1000)]`` (millisecond granularity)."""
    with wave.open(str(src), "rb") as w:
        rate, n = w.getframerate(), w.getnframes()
        a = min(int(int(start_s * 1000) * rate / 1000), n)
        b = min(int(int(end_s * 1000) * rate / 1000), n)
        w.setpos(a)
        frames = w.readframes(max(b - a, 0))
        params = w.getparams()
    with wave.open(str(dst), "wb") as o:
        o.setparams(params)
        o.writeframes(frames)
    return dst
