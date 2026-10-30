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


def test_pitch_gender_per_turn_and_lumped_speakers(tmp_path, monkeypatch):
    """Same diarization label, different voices -> per-turn gender (paper rule cannot do this)."""
    import wave

    import numpy as np

    from profanity_diarization.config import Settings
    from profanity_diarization.gender import classify, median_f0
    from profanity_diarization.pipeline import Pipeline
    from profanity_diarization.schemas import Turn

    sr = 16000

    def tone(f, s=1.0):
        t = np.arange(int(sr * s)) / sr
        return sum(np.sin(2 * np.pi * f * k * t) / k for k in (1, 2, 3)) * 0.2

    assert classify(*median_f0(tone(110), sr)) == "MALE"
    assert classify(*median_f0(tone(220), sr)) == "FEMALE"
    assert classify(*median_f0(np.zeros(sr), sr)) == "UNKNOWN"

    audio = np.concatenate([tone(110), tone(220), tone(110)])
    wav = tmp_path / "in.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(sr)
        w.writeframes((audio * 32767).astype("<i2").tobytes())
    monkeypatch.setattr(
        "profanity_diarization.pipeline.extract_audio",
        lambda s, d: d.write_bytes(wav.read_bytes()) or d,
    )

    class FakeTranscriber:
        def __init__(self, texts):
            self.texts = list(texts)

        def transcribe_file(self, w):
            return self.texts.pop(0)

    class OneSpeaker:  # a diarizer that lumps everyone into one label
        def diarize(self, w):
            return [Turn(speaker="SPEAKER_00", start=i, end=i + 1) for i in range(3)]

    pipe = Pipeline(Settings(speaker_labels="pitch_gender", lemmatize=False),
                    FakeTranscriber(["a", "oh shit", "b"]), OneSpeaker(), Detector(LEX))  # fmt: skip
    r = pipe.process(wav, tmp_path / "o")
    assert [s.group for s in r.segments] == ["MALE", "FEMALE", "MALE"]
    assert r.counts["FEMALE"] == 1 and r.counts["MALE"] == 0
    md = pipe.write_outputs(r, tmp_path / "o", "in")[3].read_text()
    assert "## Turn audit" in md and "SPEAKER_00 | 3 |" in md


def test_smooth_by_speaker_and_gender_score(tmp_path):
    from eval.gender_eval import score

    from profanity_diarization.gender import smooth_by_speaker

    g = ["MALE"] * 9 + ["FEMALE"] + ["FEMALE", "MALE", "FEMALE", "MALE"]
    s = ["A"] * 10 + ["B"] * 4
    out = smooth_by_speaker(g, s)
    assert out[:10] == ["MALE"] * 10  # 90% agree -> smoothed
    assert out[10:] == g[10:]  # 50/50 speaker is left per-turn
    assert smooth_by_speaker(["UNKNOWN", "MALE"], ["A", "A"]) == ["MALE", "MALE"]

    f = tmp_path / "t.csv"
    f.write_text("clip,turn,paper_rule,pitch_per_turn,pitch_smoothed,true_gender\n"
                 "x,0,F,M,M,M\nx,1,F,?,M,M\nx,2,M,F,F,\nx,3,M,M,M,F\n", encoding="utf-8")  # fmt: skip
    r = score(f)["methods"]
    assert r["paper_rule"]["correct"] == 0 and r["paper_rule"]["decided"] == 3
    assert r["pitch_per_turn"]["abstained"] == 1 and r["pitch_per_turn"]["correct"] == 1
    assert r["pitch_smoothed"]["correct"] == 2


def test_diar_eval_purity_and_gender():
    from eval.diar_eval import evaluate

    truth = [(0, 2, "MALE"), (3, 5, "FEMALE")]
    segs = [
        {"speaker": "A", "start": 0, "end": 2},
        {"speaker": "A", "start": 3, "end": 5},
    ]  # lumped
    r = evaluate(segs, {"m": ["MALE", "MALE"]}, truth)
    assert r["speakers_found"] == 1 and r["purity"] == 0.5
    assert r["gender"]["m"]["time_weighted_accuracy"] == 0.5


def test_load_dotenv_does_not_override_existing(tmp_path, monkeypatch):
    from profanity_diarization.cli import load_dotenv

    f = tmp_path / ".env"
    f.write_text('# c\nHF_TOKEN=abc\nOTHER="x y"\nKEEP=new\n', encoding="utf-8")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("OTHER", raising=False)
    monkeypatch.setenv("KEEP", "old")
    load_dotenv(f)
    import os

    assert (
        os.environ["HF_TOKEN"] == "abc"
        and os.environ["OTHER"] == "x y"
        and os.environ["KEEP"] == "old"
    )
    load_dotenv(tmp_path / "missing.env")  # no file: no error


def test_report_ignores_non_dataset_json_and_renders_diarization(tmp_path):
    d = {"A": {"speakers_found": 2, "speakers_true": 2, "turns": 25, "purity": 1.0,
               "gender": {"paper_rule": {"time_weighted_accuracy": 0.0},
                          "pitch_per_turn": {"time_weighted_accuracy": 1.0}}}}  # fmt: skip
    (tmp_path / "diarization_synthetic.json").write_text(json.dumps(d))
    md = render(tmp_path)
    assert "Diarizer comparison" in md and "| A | 2 / 2 | 25 | 100.0% | 0.0% | 100.0% |" in md
