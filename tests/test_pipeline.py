import json
import wave
from pathlib import Path

import pytest
from typer.testing import CliRunner

from profanity_diarization import cli
from profanity_diarization.audio import AudioError, extract_audio, ffmpeg_cmd, slice_wav
from profanity_diarization.config import Settings, load_settings
from profanity_diarization.pipeline import Pipeline
from profanity_diarization.profanity import Detector, load_lexicon, normalize
from profanity_diarization.report import srt_time
from profanity_diarization.schemas import Segment, Turn
from profanity_diarization.speakers import SpeakerGrouper

LEX = frozenset({"fuck", "shit", "son of a bitch"})
# fake "lemmatizer": strips a trailing -ed/-ing like WordNet verb lemmas would for these words
LEMMA = {"fucked": "fuck", "fucking": "fuck"}.get


def lemma(w):
    return LEMMA(w, w)


def seg(text, spk="SPEAKER_00", group="MALE", a=0.0, b=1.0):
    return Segment(speaker=spk, group=group, start=a, end=b, text=text)


def make_wav(path: Path, seconds=3, rate=16000):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x01\x00" * rate * seconds)
    return path


class FakeDiarizer:
    calls = 0

    def diarize(self, wav):
        FakeDiarizer.calls += 1
        return [Turn(speaker="SPEAKER_00", start=0, end=1), Turn(speaker="SPEAKER_01", start=1, end=2),
                Turn(speaker="SPEAKER_00", start=2, end=3)]  # fmt: skip


class FakeTranscriber:
    texts = iter([])

    def __init__(self, texts):
        self.texts, self.calls = list(texts), 0

    def transcribe_file(self, wav):
        self.calls += 1
        return self.texts.pop(0)


def test_bundled_lexicons():
    assert len(load_lexicon()) == 20 and "fuck" in load_lexicon()
    assert load_lexicon("lexicon_a") != load_lexicon("lexicon_b")
    with pytest.raises(FileNotFoundError):
        load_lexicon("nope")


def test_matching_rule():
    d = Detector(LEX, lemma)
    assert d.match("Fuck!") == "fuck"  # case + punctuation
    assert d.match("f.u.c.k") == "fuck"  # ALL punctuation removed, as legacy
    assert d.match("fucked,") == "fuck"  # lemma
    assert d.match("fucker") is None  # no substring matching
    assert d.match("son") is None
    assert normalize("Sh!t") == "sht"


def test_multiword_entry_never_matches():  # documented legacy bug, kept in `paper`
    hits = Detector(LEX).detect([seg("you son of a bitch")])
    assert hits == []


def test_repeated_tokens_counted_with_turn_times():
    hits = Detector(LEX).detect([seg("shit shit well shit", a=4, b=6)])
    assert [h.term for h in hits] == ["shit"] * 3
    assert (hits[0].start, hits[0].end) == (4, 6)


def test_mask():
    assert Detector(LEX).mask("Oh, shit! ok") == "Oh, s*** ok"


def test_speaker_grouper_paper_heuristic():
    g = SpeakerGrouper()
    assert [g(s) for s in ("SPEAKER_00", "SPEAKER_01", "SPEAKER_02", "SPEAKER_01")] == [
        "MALE", "FEMALE", "MALE", "FEMALE"]  # fmt: skip
    assert SpeakerGrouper()("SPEAKER_00") == "MALE"
    assert SpeakerGrouper("raw")("SPEAKER_07") == "SPEAKER_07"


def test_srt_time():
    assert srt_time(3661.5) == "01:01:01,500"
    assert srt_time(0.0004) == "00:00:00,000"


def test_ffmpeg_cmd_handles_spaces_unicode(tmp_path):
    cmd = ffmpeg_cmd(Path("my vidéo 日本.mp4"), tmp_path / "o u t.wav")
    assert cmd[3] == "my vidéo 日本.mp4" and cmd[-1].endswith("o u t.wav")
    assert "-ar" in cmd and cmd[cmd.index("-ar") + 1] == "16000"


def test_extract_audio_errors(tmp_path):
    with pytest.raises(AudioError, match="not found"):
        extract_audio(tmp_path / "missing.mp4", tmp_path / "o.wav")


def test_slice_wav_ms_granularity(tmp_path):
    src = make_wav(tmp_path / "ünï code.wav", seconds=3)
    out = slice_wav(src, 1.0, 2.5, tmp_path / "s.wav")
    with wave.open(str(out)) as w:
        assert w.getnframes() == 24000
    out = slice_wav(src, 2.5, 99, tmp_path / "s2.wav")  # end beyond file is clamped
    with wave.open(str(out)) as w:
        assert w.getnframes() == 8000


def test_presets():
    p = load_settings("paper")
    assert (p.whisper_model, p.language, p.lemmatize) == ("small", "en", True)
    assert p.diarization_model == "pyannote/speaker-diarization"
    assert load_settings("legacy_v1").lemmatize is False
    with pytest.raises(FileNotFoundError):
        load_settings("nope")


def make_pipe(cfg=None):
    cfg = cfg or Settings(lemmatize=False)
    tr = FakeTranscriber(["hello there", "oh shit", ""])
    return Pipeline(cfg, tr, FakeDiarizer(), Detector(LEX)), tr


def test_pipeline_end_to_end_and_cache(tmp_path, monkeypatch):
    src = make_wav(tmp_path / "in.wav")
    monkeypatch.setattr("profanity_diarization.pipeline.extract_audio",
                        lambda s, d: make_wav(d))  # fmt: skip
    pipe, tr = make_pipe()
    r = pipe.process(src, tmp_path / "out")
    assert [s.text for s in r.segments] == ["hello there", "oh shit"]  # empty turn dropped
    assert r.counts == {"MALE": 0, "FEMALE": 1} and r.unique_counts["FEMALE"] == 1
    assert r.hits[0].speaker == "SPEAKER_01" and r.hits[0].start == 1
    assert tr.calls == 3
    # second run: diarize + asr served from cache, no model calls
    FakeDiarizer.calls = 0
    pipe2, tr2 = make_pipe()
    r2 = pipe2.process(src, tmp_path / "out")
    assert tr2.calls == 0 and FakeDiarizer.calls == 0
    assert r2.segments == r.segments


def test_outputs_masked_and_schema(tmp_path, monkeypatch):
    src = make_wav(tmp_path / "in.wav")
    monkeypatch.setattr("profanity_diarization.pipeline.extract_audio", lambda s, d: make_wav(d))
    pipe, _ = make_pipe()
    r = pipe.process(src, tmp_path / "o")
    j, c, s, m = pipe.write_outputs(r, tmp_path / "o", "in")
    assert json.loads(j.read_text())["counts"]["FEMALE"] == 1
    assert c.read_text().splitlines()[0] == "term,token,speaker,group,start,end,context"
    assert "shit" in c.read_text() and "shit" not in s.read_text() and "shit" not in m.read_text()
    assert "2\n00:00:01,000 --> 00:00:02,000\n[SPEAKER_01 | FEMALE] oh s***" in s.read_text()


def test_cli_with_fakes(tmp_path, monkeypatch):
    src = make_wav(tmp_path / "clip.wav")
    monkeypatch.setattr("profanity_diarization.pipeline.extract_audio", lambda s, d: make_wav(d))
    monkeypatch.setattr(cli, "Pipeline", lambda cfg: make_pipe(cfg)[0])
    res = CliRunner().invoke(cli.app, ["process", str(src), "--output-dir", str(tmp_path / "o")])
    assert res.exit_code == 0, res.output
    assert (tmp_path / "o" / "clip.json").exists()
    res = CliRunner().invoke(cli.app, ["process", str(tmp_path / "nope.wav")])
    assert res.exit_code == 1


def test_cli_batch_empty_dir_fails(tmp_path):
    assert CliRunner().invoke(cli.app, ["batch", str(tmp_path)]).exit_code == 1
