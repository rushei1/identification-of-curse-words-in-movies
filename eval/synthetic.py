"""Synthetic sanity set: TTS sentences with known words (labels by construction).

SYNTHETIC and not representative of real movies: clean TTS speech, two voices, scripted words.
Words prefixed with ``*`` are the positives; every other token is negative. Negatives include
innocent words that contain a profane stem and lexicon words used literally, so the set
stresses precision as well as recall.

  python -m eval.synthetic audio    # -> eval/data/synthetic.wav  (Windows System.Speech, offline)
  python -m eval.synthetic labels   # report JSON -> eval/data/synthetic.labels.json
"""

from __future__ import annotations

import json
import subprocess
import sys
import wave
from pathlib import Path

from profanity_diarization.profanity import normalize

DATA = Path(__file__).parent / "data"
D, Z = "Microsoft David Desktop", "Microsoft Zira Desktop"
SENTENCES = [
    (D, "Oh *shit, I forgot my keys again."),
    (D, "This is a load of *bullshit and everybody knows it."),
    (D, "He called me a *bastard in front of the whole class."),
    (D, "Get the *fuck out of my office right now."),
    (D, "That was a *damn good game last night."),
    (D, "I screwed the lid on the jar and put the nuts in the cupboard."),  # literal use, negative
    (D, "It was a freak accident on the highway."),  # literal use, negative
    (D, "The engineers visited Scunthorpe and met Charles Dickens fans."),  # stem inside innocent word
    (D, "My assistant will pass the message to the class."),
    (D, "The vacuum sucks up the dust very quickly."),  # literal use, negative
    (D, "Those *motherfuckers stole my wallet."),
    (D, "She is such a *bitch sometimes."),
    (D, "He read Moby Dick in high school."),  # a name: ambiguous for any lexical matcher
    (D, "You are a stupid *asshole and I hate you."),
    (D, "Go to *hell and take your friends with you."),
    (Z, "Oh *crap, the bus just left without us."),
    (Z, "What the *fuck was that noise outside?"),
    (Z, "He is a lazy *dickhead who never helps."),
    (Z, "Those *bitches ruined the whole party."),
    (Z, "The shiitake mushrooms were delicious and the shell was thin."),
    (Z, "I will not *piss away my savings on that."),
    (Z, "Pissarro painted beautiful landscapes."),
    (Z, "Hello everyone, welcome to the show, we will begin the passage reading."),
    (Z, "That guy is a total *jerk."),
    (Z, "That was a really dumb idea."),  # mild lexicon word, non-vulgar use: negative
]


def _speak(voice: str, text: str, out: Path) -> None:
    ps = (
        "Add-Type -AssemblyName System.Speech; $s=New-Object System.Speech.Synthesis.SpeechSynthesizer;"
        f"$s.SelectVoice('{voice}'); $s.SetOutputToWaveFile('{out}'); $s.Speak('{text}'); $s.Dispose()"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)


def make_audio(out: Path = DATA / "synthetic.wav") -> Path:
    tmp = DATA / "synth"
    tmp.mkdir(parents=True, exist_ok=True)
    pcm, params = b"", None
    for i, (voice, text) in enumerate(SENTENCES):
        raw, wav = tmp / f"{i}.raw.wav", tmp / f"{i}.wav"
        _speak(voice, text.replace("*", ""), raw)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-ar", "16000",
                        "-ac", "1", str(wav)], check=True)  # fmt: skip
        with wave.open(str(wav)) as w:
            params = w.getparams()
            pcm += w.readframes(w.getnframes()) + b"\x00\x00" * 16000  # 1 s of silence
    with wave.open(str(out), "wb") as o:
        o.setparams(params)
        o.writeframes(pcm)
    return out


def gold_forms() -> set:
    return {normalize(w) for _, t in SENTENCES for w in t.split() if w.startswith("*")}


def make_labels(report: Path, out: Path = DATA / "synthetic.labels.json") -> dict:
    r = json.loads(report.read_text(encoding="utf-8"))
    forms, pos, seen = gold_forms(), [], set()
    for si, s in enumerate(r["segments"]):
        for ti, t in enumerate(s["text"].split()):
            n = normalize(t)
            if n in forms:
                pos.append(f"{si}:{ti}")
                seen.add(n)
    n_marked = sum(w.startswith("*") for _, t in SENTENCES for w in t.split())
    spec = {
        "report": str(Path("..") / ".." / report) if not report.is_absolute() else str(report),
        "labeler": "by construction (scripted TTS text; known positives)",
        "notes": f"SYNTHETIC, not representative of real movies. {n_marked} scripted profane words; "
                 f"{len(pos)} recognised verbatim by ASR in the transcript, {len(forms - seen)} of "
                 f"{len(forms)} distinct forms never appeared (ASR miss, not a detector miss).",
        "positives": pos,
    }  # fmt: skip
    out.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    return spec


if __name__ == "__main__":
    if sys.argv[1:] == ["audio"]:
        print(make_audio())
    elif sys.argv[1:2] == ["labels"]:
        print(make_labels(Path(sys.argv[2]))["notes"])
    else:
        sys.exit(__doc__)
