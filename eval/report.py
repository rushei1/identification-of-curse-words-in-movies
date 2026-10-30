"""Render eval/results/*.json -> eval/results.md (terms masked; includes a 'Not run' section)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent
EXPECTED = {
    "sample-video": "real clip 1 (token labels)",
    "newyorker": "real clip 2 (token labels)",
    "synthetic": "synthetic TTS sentences (labels by construction)",
}


def pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def table(res: dict) -> list:
    L = ["| Preset | TP | FP | FN | TN | Accuracy | Precision | Recall | F1 | Specificity | FPR | FNR |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for p, v in res["presets"].items():
        L.append(f"| {p} | {v['tp']} | {v['fp']} | {v['fn']} | {v['tn']} | {pct(v['accuracy'])} | "
                 f"{pct(v['precision'])} | {pct(v['recall'])} | {pct(v['f1'])} | "
                 f"{pct(v['specificity'])} | {pct(v['false_positive_rate'])} | "
                 f"{pct(v['false_negative_rate'])} |")  # fmt: skip
    return L


REAL = ("sample-video", "newyorker")


def pooled(res: dict) -> dict:
    """Micro-average over the real clips: sum the confusion matrices, then rescore."""
    from .metrics import scores

    names = [n for n in REAL if n in res]
    out = {"n_tokens": 0, "n_gold_positive": 0, "presets": {}}
    for n in names:
        out["n_tokens"] += res[n]["n_tokens"]
        out["n_gold_positive"] += res[n]["n_gold_positive"]
    for p in res[names[0]]["presets"]:
        c = {k: sum(res[n]["presets"][p][k] for n in names) for k in ("tp", "fp", "fn", "tn")}
        out["presets"][p] = scores(c)
    out["names"] = names
    return out


def render(results_dir: Path = ROOT / "results") -> str:
    allj = {p.stem: json.loads(p.read_text()) for p in sorted(results_dir.glob("*.json"))}
    res = {k: v for k, v in allj.items() if "presets" in v}  # token-level dataset results
    L = ["# Evaluation results", "",
         "Unit: **token** (whitespace-split ASR output); *profane* is the positive class. Accuracy is "
         "dominated by true negatives (most words are not profane), so read precision, recall and F1 "
         "first. All profane terms are masked.", ""]  # fmt: skip
    if res:
        m = next(iter(res.values())).get("meta", {})
        L += [f"Run: {m.get('date', '?')} | Python {m.get('python', '?')} | {m.get('platform', '?')}", ""]
    real = [n for n in REAL if n in res]
    if len(real) > 1:
        pl = pooled(res)
        L += [f"## Headline: real clips pooled ({' + '.join(real)})", "",
              f"{pl['n_tokens']} tokens, {pl['n_gold_positive']} gold-positive (micro-average: confusion "
              "matrices summed, then rescored). Labels are model-assigned, not independent human "
              "annotation; the `improved` matcher was designed from the first clip, so only the second "
              "clip and the synthetic set are out-of-sample for it.", ""]  # fmt: skip
        L += table(pl) + [""]
    for name, r in res.items():
        L += [f"## {name}", "",
              f"{r['n_tokens']} tokens, {r['n_gold_positive']} gold-positive. Labels by: {r['labeler']}. "
              f"{r['notes']}", ""]  # fmt: skip
        L += table(r) + [""]
        t = r.get("timings_s") or {}
        if t.get("diarize", 0) >= 1:
            L += ["Stage time (CPU): " + ", ".join(f"{k} {v:.0f}s" for k, v in t.items()), ""]
        else:
            L += ["Stage time: not recorded (report rebuilt from the stage cache).", ""]
        for p, v in r["presets"].items():
            fp, fn = Counter(v["false_positives"]), Counter(v["false_negatives"])
            L.append(f"- **{p}** false positives: {dict(fp) or 'none'}; false negatives: {dict(fn) or 'none'}")
        L.append("")
    L += ["## Reading the errors", "",
          "- **False negatives** are words outside the 20-term lexicon or in a form it cannot match "
          "(compounds, agent nouns such as the *-er* form of a verb). Exact-match `paper` cannot catch them.",
          "- **False positives** are lexicon words used in a mild or literal sense (`screwed`, `freak`, "
          "`dumb`, `heck`) and names (a surname or a book title). Neither preset uses context, so "
          "`improved` does not reduce them.",
          "- Gold labels cover **detection on the ASR output**. Words Whisper mis-hears or drops are "
          "invisible to this metric; the synthetic set reports how many scripted words ASR recognised.",
          "- Token-level counts differ from the paper's hit counting only for multi-word phrases.", ""]
    if "diarization_synthetic" in allj:
        L += ["## Diarizer comparison on the synthetic clip (truth known)", "",
              "| Diarizer | Speakers found / true | Turns | Purity | Gender: paper rule | Gender: pitch per turn |",
              "|---|---|---|---|---|---|"]  # fmt: skip
        for name, v in allj["diarization_synthetic"].items():
            g = v["gender"]
            L.append(f"| {name} | {v['speakers_found']} / {v['speakers_true']} | {v['turns']} | "
                     f"{pct(v['purity'])} | {pct(g['paper_rule']['time_weighted_accuracy'])} | "
                     f"{pct(g['pitch_per_turn']['time_weighted_accuracy'])} |")  # fmt: skip
        L.append("")
    nr = [f"- **{k}**: {d}: not run (no results file)" for k, d in EXPECTED.items() if k not in res]
    L += ["## Not run", ""] + (nr or ["Nothing skipped."]) + [""]
    # brief rule: no lexicon term appears unmasked in any doc (also masks words quoted in the notes)
    from profanity_diarization.config import Settings
    from profanity_diarization.profanity import build_detector

    det = build_detector(Settings(lemmatize=False, match="improved"))  # no WordNet needed
    return "\n".join(det.mask(line) for line in L)


def main():
    (ROOT / "results.md").write_text(render(), encoding="utf-8")
    print(ROOT / "results.md")


if __name__ == "__main__":
    main()
