"""Speaker-gender evaluation against hand labels (the paper's SPEAKER_01 rule vs pitch variants).

  python -m eval.gender_eval template   # writes eval/data/gender_template.csv  (fill true_gender: M or F)
  python -m eval.gender_eval score      # scores every filled row -> eval/results/gender.json
Only turns you label are scored; leave true_gender empty to skip a turn.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from profanity_diarization.config import Settings
from profanity_diarization.gender import smooth_by_speaker
from profanity_diarization.profanity import build_detector

ROOT = Path(__file__).parent
CLIPS = {
    "netflix": "*Netflix",
    "newyorker": "*New Yorker",
}  # name -> report stem glob (user file names)
METHODS = ("paper_rule", "pitch_per_turn", "pitch_smoothed")
SHORT = {"MALE": "M", "FEMALE": "F", "UNKNOWN": "?"}


def load(folder, pattern):
    (path,) = (ROOT.parent / folder).glob(pattern + ".json")
    return json.loads(path.read_text(encoding="utf-8"))


def rows():
    det = build_detector(Settings(lemmatize=False, match="improved"))  # masks only
    for clip, pat in CLIPS.items():
        paper, imp = load("output", pat), load("output_improved", pat)
        spk = [s["speaker"] for s in imp["segments"]]
        per_turn = [s["group"] for s in imp["segments"]]
        smooth = smooth_by_speaker(per_turn, spk)
        for i, s in enumerate(imp["segments"]):
            assert s["start"] == paper["segments"][i]["start"]  # same turns in both reports
            yield {"clip": clip, "turn": i, "start": round(s["start"], 1), "end": round(s["end"], 1),
                   "speaker": s["speaker"], "f0": s["f0"], "snippet": det.mask(" ".join(s["text"].split()[:7])),
                   "paper_rule": SHORT[paper["segments"][i]["group"]],
                   "pitch_per_turn": SHORT[per_turn[i]], "pitch_smoothed": SHORT[smooth[i]],
                   "true_gender": ""}  # fmt: skip


def template():
    out = ROOT / "data" / "gender_template.csv"
    rs = list(rows())
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rs[0]))
        w.writeheader()
        w.writerows(rs)
    print(out, len(rs), "turns")


def score(path: Path = ROOT / "data" / "gender_template.csv") -> dict:
    labeled = [
        r
        for r in csv.DictReader(open(path, encoding="utf-8"))
        if r["true_gender"].strip().upper() in ("M", "F")
    ]
    res = {"n_labeled": len(labeled), "methods": {}}
    for m in METHODS:
        decided = [r for r in labeled if r[m] in ("M", "F")]
        correct = sum(r[m] == r["true_gender"].strip().upper() for r in decided)
        res["methods"][m] = {"correct": correct, "decided": len(decided), "abstained": len(labeled) - len(decided),
                             "accuracy_on_decided": correct / len(decided) if decided else None,
                             "accuracy_all_labeled": correct / len(labeled) if labeled else None}  # fmt: skip
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "gender.json").write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    {"template": template, "score": lambda: print(json.dumps(score(), indent=2))}[sys.argv[1]]()
