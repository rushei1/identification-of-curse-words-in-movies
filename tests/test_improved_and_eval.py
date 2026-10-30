import json

import pytest

from eval.metrics import confusion, scores
from eval.report import render
from profanity_diarization.config import load_settings
from profanity_diarization.profanity import Detector, load_lexicon
from profanity_diarization.schemas import Segment

STEMS, STOP = load_lexicon("stems"), load_lexicon("stoplist")
LEX = load_lexicon()


def det(**kw):
    return Detector(LEX, stems=STEMS, stoplist=STOP, **kw)


def seg(text):
    return Segment(speaker="S", group="MALE", start=0, end=1, text=text)


def test_paper_matcher_unchanged_by_new_options():
    d = Detector(LEX)
    assert d.match("horseshit") is None and d.match("motherfuckers!") is None
    assert d.match("Shit.") == "shit"


def test_stems_catch_compounds_and_plurals():
    d = det()
    for w in ("horseshit", "Motherfuckers!", "bullshit", "dickhead", "cunt."):
        assert d.match(w), w


def test_stoplist_blocks_innocent_words():
    d = det()
    for w in ("Scunthorpe", "Dickens", "Pissarro", "shitake", "class", "assistant", "hello"):
        assert d.match(w) is None, w


def test_phrase_counts_once():
    hits = det(phrases=True).detect([seg("you son of a bitch")])
    assert [h.term for h in hits] == ["son of a bitch"]
    # without phrase mode the stem still flags the head word
    assert [h.term for h in det().detect([seg("you son of a bitch")])] == ["bitch"]


def test_improved_preset_loads():
    assert load_settings("improved").match == "improved"
    assert load_settings("paper").match == "exact"


def test_metrics():
    c = confusion([1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 0, 1, 0, 0, 0, 0])
    assert c == {"tp": 2, "fp": 1, "fn": 1, "tn": 4}
    s = scores(c)
    assert s["precision"] == pytest.approx(2 / 3) and s["recall"] == pytest.approx(2 / 3)
    assert s["f1"] == pytest.approx(2 / 3) and s["accuracy"] == pytest.approx(6 / 8)
    assert s["false_positive_rate"] == pytest.approx(1 / 5)


def test_metrics_undefined_not_zero():
    s = scores(confusion([0, 0], [0, 0]))
    assert s["precision"] is None and s["recall"] is None and s["f1"] is None


def test_report_not_run_and_masking(tmp_path):
    assert "not run (no results file)" in render(tmp_path)  # empty dir: everything skipped
    r = {"dataset": "synthetic", "n_tokens": 4, "n_gold_positive": 1, "labeler": "x", "notes": "",
         "meta": {}, "presets": {"paper": {**scores(confusion([1, 0], [0, 1])),
                                           "false_positives": ["s***"], "false_negatives": ["f***"]}}}  # fmt: skip
    (tmp_path / "synthetic.json").write_text(json.dumps(r))
    md = render(tmp_path)
    assert "synthetic" in md and "n/a" in md and "**synthetic**: " not in md.split("## Not run")[1]


def test_pooled_sums_confusions():
    from eval.report import pooled

    def r(tp, fp, fn, tn):
        return {"n_tokens": tp + fp + fn + tn, "n_gold_positive": tp + fn,
                "presets": {"paper": scores({"tp": tp, "fp": fp, "fn": fn, "tn": tn})}}  # fmt: skip

    p = pooled({"sample-video": r(10, 0, 3, 261), "newyorker": r(28, 3, 4, 812)})
    v = p["presets"]["paper"]
    assert (v["tp"], v["fp"], v["fn"], v["tn"]) == (38, 3, 7, 1073)
    assert v["precision"] == pytest.approx(38 / 41)


def test_report_masks_lexicon_words_in_notes(tmp_path):
    r = {"dataset": "x", "n_tokens": 1, "n_gold_positive": 0, "labeler": "t",
         "notes": "mild words like dumb and sucks, plus shit", "meta": {},
         "presets": {"paper": {**scores(confusion([0], [0])), "false_positives": [],
                               "false_negatives": []}}}  # fmt: skip
    (tmp_path / "x.json").write_text(json.dumps(r))
    md = render(tmp_path)
    assert "shit" not in md and "dumb" not in md and "sucks" not in md and "s***" in md
