"""Speaker label -> group. ``paper_gender`` is the legacy label heuristic, NOT gender detection."""

from __future__ import annotations


class SpeakerGrouper:
    def __init__(self, mode: str = "paper_gender"):
        self.mode, self.map, self.female = mode, {}, 0

    def __call__(self, speaker: str) -> str:
        if self.mode == "raw":
            return speaker
        if speaker not in self.map:  # final_gpu.py:74-80, first-seen order
            if "SPEAKER_01" in speaker and self.female == 0:
                self.map[speaker] = "FEMALE"
                self.female += 1
            else:
                self.map[speaker] = "MALE"
        return self.map[speaker]
