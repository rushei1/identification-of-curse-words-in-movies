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


def render(results_dir: Path = ROOT / "results") -> str:
    res = {p.stem: json.loads(p.read_text()) for p in sorted(results_dir.glob("*.json"))}
    L = ["# Evaluation results", "",
         "Unit: **token** (whitespace-split ASR output); *profane* is the positive class. Accuracy is "
         "dominated by true negatives (most words are not profane), so read precision, recall and F1 "
         "first. All profane terms are masked.", ""]  # fmt: skip
    if res:
        m = next(iter(res.values())).get("meta", {})
        L += [f"Run: {m.get('date', '?')} | Python {m.get('python', '?')} | {m.get('platform', '?')}", ""]
    for name, r in res.items():
        L += [f"## {name}", "",
              f"{r['n_tokens']} tokens, {r['n_gold_positive']} gold-positive. Labels by: {r['labeler']}. "
              f"{r['notes']}", ""]  # fmt: skip
        L += table(r) + [""]
        for p, v in r["presets"].items():
            fp, fn = Counter(v["false_positives"]), Counter(v["false_negatives"])
            L.append(f"- **{p}** false positives: {dict(fp) or 'none'}; false negatives: {dict(fn) or 'none'}")
        L.append("")
    nr = [f"- **{k}**: {d}: not run (no results file)" for k, d in EXPECTED.items() if k not in res]
    L += ["## Not run", ""] + (nr or ["Nothing skipped."]) + [""]
    return "\n".join(L)


def main():
    (ROOT / "results.md").write_text(render(), encoding="utf-8")
    print(ROOT / "results.md")


if __name__ == "__main__":
    main()
