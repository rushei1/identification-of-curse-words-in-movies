"""Diarization + gender quality on the SYNTHETIC clip, where the truth is known by construction.

Truth timeline = the scripted sentences (eval/data/synth/<i>.wav durations + 1 s silence);
sentences 0-14 are the male voice, 15-24 the female voice. Compares diarizer x gender method.
"""

from __future__ import annotations

import json
import wave
from collections import defaultdict
from pathlib import Path

from profanity_diarization.speakers import SpeakerGrouper

ROOT = Path(__file__).parent
REPORTS = {  # diarizer -> (paper-preset report, pitch-gender report)
    "pyannote/speaker-diarization (paper)": ("output/synthetic.json", "output_improved/synthetic.json"),
    "pyannote/speaker-diarization-3.1": (None, "output_31/synthetic.json"),
}  # fmt: skip


def truth_intervals(n=25, split=15, gap=1.0):
    t, out = 0.0, []
    for i in range(n):
        with wave.open(str(ROOT / "data" / "synth" / f"{i}.wav")) as w:
            d = w.getnframes() / w.getframerate()
        out.append((t, t + d, "MALE" if i < split else "FEMALE"))
        t += d + gap
    return out


def overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def evaluate(segments: list, groups_by_method: dict, truth: list) -> dict:
    """Duration-weighted. ``segments``: dicts with speaker/start/end. Truth voice of a turn = max overlap."""
    contingency = defaultdict(lambda: defaultdict(float))
    rows = []
    for k, s in enumerate(segments):
        ov = defaultdict(float)
        for t0, t1, v in truth:
            ov[v] += overlap(s["start"], s["end"], t0, t1)
        if not ov or max(ov.values()) == 0:
            continue
        voice, w = max(ov.items(), key=lambda x: x[1])
        contingency[s["speaker"]][voice] += w
        rows.append((k, voice, w))
    total = sum(w for _, _, w in rows)
    res = {"speakers_found": len({s["speaker"] for s in segments}), "speakers_true": 2, "turns": len(segments),
           "purity": sum(max(c.values()) for c in contingency.values()) / total if total else None,
           "gender": {}}  # fmt: skip
    for m, groups in groups_by_method.items():
        ok = sum(w for k, v, w in rows if groups[k] == v)
        res["gender"][m] = {"time_weighted_accuracy": ok / total if total else None,
                            "unknown_share": sum(w for k, v, w in rows if groups[k] == "UNKNOWN") / total}  # fmt: skip
    return res


def main():
    truth, out = truth_intervals(), {}
    for name, (_, pitch_p) in REPORTS.items():
        imp = json.loads((ROOT.parent / pitch_p).read_text(encoding="utf-8"))
        segs = imp["segments"]
        g = SpeakerGrouper()
        methods = {"paper_rule": [g(s["speaker"]) for s in segs],
                   "pitch_per_turn": [s["group"] for s in segs]}  # fmt: skip
        out[name] = evaluate(segs, methods, truth)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "diarization_synthetic.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
