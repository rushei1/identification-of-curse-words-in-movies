"""Score detector presets on cached transcripts against token-level gold labels.

Labels file ``eval/data/<name>.labels.json``:
  {"report": "<path to a report JSON>", "positives": ["seg:tok", ...], "labeler": "...", "notes": "..."}
``seg:tok`` indexes ``report.segments[seg].text.split()[tok]``. Everything not listed is negative.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from profanity_diarization.config import load_settings
from profanity_diarization.profanity import build_detector
from profanity_diarization.schemas import Report

from .metrics import confusion, scores

ROOT = Path(__file__).parent
PRESETS = ["paper", "improved"]


def tokens(report: Report):
    for si, s in enumerate(report.segments):
        for ti, t in enumerate(s.text.split()):
            yield si, ti, t


def mask(t: str) -> str:
    t = "".join(ch for ch in t if ch.isalnum() or ch == "'")
    return (t[:1] + "***") if t else "***"


def evaluate(labels_path: Path, presets=PRESETS) -> dict:
    spec = json.loads(labels_path.read_text(encoding="utf-8"))
    rep_path = (labels_path.parent / spec["report"]).resolve()
    report = Report.model_validate_json(rep_path.read_text(encoding="utf-8"))
    toks = list(tokens(report))
    if (
        "forms_from" in spec
    ):  # other diarizer/ASR segmentation: label by the gold word FORMS of a labeled run
        from profanity_diarization.profanity import normalize

        base = json.loads((labels_path.parent / spec["forms_from"]).read_text(encoding="utf-8"))
        brep = Report.model_validate_json(
            (labels_path.parent / base["report"]).read_text(encoding="utf-8")
        )
        bt = {f"{a}:{b}": t for a, b, t in tokens(brep)}
        forms = {normalize(bt[k]) for k in base["positives"]}
        gold = [normalize(t) in forms for _, _, t in toks]
        gold_set = {k for k, g in zip((f"{a}:{b}" for a, b, _ in toks), gold) if g}
    else:
        gold_set = set(spec["positives"])
        gold = [f"{s}:{t}" in gold_set for s, t, _ in toks]
        assert sum(gold) == len(gold_set), "labels refer to tokens that do not exist"
    out = {"dataset": labels_path.name.replace(".labels.json", ""), "n_tokens": len(toks),
           "n_gold_positive": sum(gold), "labeler": spec.get("labeler", "unknown"),
           "notes": spec.get("notes", ""), "timings_s": report.timings, "presets": {}}  # fmt: skip
    for p in presets:
        det = build_detector(load_settings(p))
        pred = [det.match(t) is not None for _, _, t in toks]
        res = scores(confusion(gold, pred))
        res["false_positives"] = [
            mask(t) for (_, _, t), g, q in zip(toks, gold, pred) if q and not g
        ]
        res["false_negatives"] = [
            mask(t) for (_, _, t), g, q in zip(toks, gold, pred) if g and not q
        ]
        out["presets"][p] = res
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("labels", nargs="*", type=Path, help="default: eval/data/*.labels.json")
    a = ap.parse_args(argv)
    files = a.labels or sorted((ROOT / "data").glob("*.labels.json"))
    meta = {"date": datetime.now(timezone.utc).isoformat(), "python": sys.version.split()[0],
            "platform": platform.platform()}  # fmt: skip
    for f in files:
        r = evaluate(f) | {"meta": meta}
        (ROOT / "results").mkdir(exist_ok=True)
        (ROOT / "results" / f"{r['dataset']}.json").write_text(json.dumps(r, indent=2))
        print(
            r["dataset"],
            {p: (v["precision"], v["recall"], v["f1"]) for p, v in r["presets"].items()},
        )


if __name__ == "__main__":
    main()
