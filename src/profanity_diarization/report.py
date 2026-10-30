"""Outputs: JSON (canonical), CSV, SRT, Markdown. SRT and Markdown mask profane terms."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from .profanity import Detector
from .schemas import Report


def srt_time(t: float) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_json(r: Report, path: Path) -> None:
    Path(path).write_text(r.model_dump_json(indent=2), encoding="utf-8")


def write_csv(r: Report, path: Path) -> None:
    cols = ["term", "token", "speaker", "group", "start", "end", "context"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for h in r.hits:
            w.writerow([getattr(h, c) for c in cols])


def write_srt(r: Report, det: Detector, path: Path) -> None:
    blocks = []
    for i, s in enumerate(r.segments, 1):
        text = det.mask(s.text)
        blocks.append(
            f"{i}\n{srt_time(s.start)} --> {srt_time(s.end)}\n[{s.speaker} | {s.group}] {text}\n"
        )
    Path(path).write_text("\n".join(blocks), encoding="utf-8")


def write_markdown(r: Report, det: Detector, path: Path) -> None:
    L = [f"# Profanity report: {Path(r.source).name}", "",
         f"Preset: `{r.preset}` | created: {r.created}", "",
         "## Per-speaker-group counts", "",
         "| Group | Total | Unique | Terms |", "|---|---|---|---|"]  # fmt: skip
    for g, n in r.counts.items():
        terms = Counter(det.mask(h.term) for h in r.hits if h.group == g)
        t = ", ".join(f"{k} x{v}" for k, v in terms.items()) or "-"
        L.append(f"| {g} | {n} | {r.unique_counts.get(g, 0)} | {t} |")
    L += ["", "## Per diarization speaker", "",
          "| Speaker | Turns | Hits | Gender per turn |", "|---|---|---|---|"]  # fmt: skip
    for spk in sorted({x.speaker for x in r.segments}):
        segs = [x for x in r.segments if x.speaker == spk]
        mix = ", ".join(f"{g} {n}" for g, n in Counter(x.group for x in segs).items())
        L.append(f"| {spk} | {len(segs)} | {sum(h.speaker == spk for h in r.hits)} | {mix} |")
    L += ["", "## Turn audit (check these against the video)", "",
          "| # | Time | Speaker | Group | F0 (Hz) | Hits |", "|---|---|---|---|---|---|"]  # fmt: skip
    for i, x in enumerate(r.segments):
        n = sum(h.start == x.start and h.speaker == x.speaker for h in r.hits)
        f0 = "-" if x.f0 is None else f"{x.f0:.0f}"
        L.append(f"| {i} | {srt_time(x.start)[:-4]}-{srt_time(x.end)[:-4]} | {x.speaker} | "
                 f"{x.group} | {f0} | {n} |")  # fmt: skip
    L += ["", "## Timeline", ""]
    if not r.hits:
        L.append("No profanity detected.")
    for h in r.hits:
        L.append(
            f"- {srt_time(h.start)}-{srt_time(h.end)} **{h.speaker}** ({h.group}): "
            f"`{det.mask(h.term)}` in \"{det.mask(h.context)}\""
        )
    L += ["", "_Timestamps are turn-level (the paper pipeline keeps no word timestamps)._", ""]
    Path(path).write_text("\n".join(L), encoding="utf-8")
