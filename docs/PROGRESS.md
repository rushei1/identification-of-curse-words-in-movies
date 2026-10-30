# Progress

Legacy dir: the author's original scattered project folder (read-only, not part of this repo). This repo was created as a subfolder of it.

## Status
- [x] Phase A audit (`AUDIT.md`); owner answered with "use defaults", HF token via `.env`, work in a venv.
- [x] Phase B core: schemas, config + presets (`paper`, `legacy_v1`, `improved`), audio, ASR, diarization, speaker grouping, profanity, reports, cached pipeline, CLI.
- [x] First real end-to-end runs of the `paper` preset (CPU), see "Measured" below.
- [x] Phase E (partial): token-level evaluation of `paper` vs `improved` on 2 real clips + 1 synthetic set -> `eval/results.md`.
- [ ] `scripts/compare_with_legacy.py` (fidelity vs the legacy script): **not run, fidelity UNVERIFIED**.
- [ ] README, Docker, CI, Makefile, CITATION, `scripts/setup_venv.ps1`, auto `.env` loading, tags `v0.1.0`/`v1.0.0`.
- [ ] Phase D rest (faster-whisper, overlap handling, censoring, batch perf), Phase E ASR-WER/DER on public data.

## Measured (real runs, CPU, Whisper `small`, pyannote.audio 3.3.2, torch 2.7.1+cpu, Python 3.9.13)
| Clip | Length | diarize | asr | Paper hits (MALE/FEMALE) |
|---|---|---|---|---|
| clip 1 (trailer) | 120 s | 360 s | 70 s | 9 / 1 |
| clip 2 (interview) | 392 s | not recorded (run was killed for low memory after caching; report rebuilt from cache) | | 27 / 4 |
| synthetic TTS | 103 s | 247 s | 107 s | 5 / 9 |

Detection metrics (token level, pooled real clips, labels model-assigned): see `eval/results.md`. Headline: `paper` precision 92.7%, recall 84.4%, F1 88.4%; `improved` precision 93.8%, recall 100%, F1 96.8% (clip 1 is in-sample for `improved`).

## Decisions
- LEGACY_DIR = parent folder of this repo; repo is a subfolder (owner's instruction).
- The brief assumes word-level ASR + alignment; legacy is diarize-then-transcribe-per-turn. Trusting the code (brief rule 9): `paper` follows the legacy flow; hit timestamps are turn-level.
- Presets are packaged in `src/profanity_diarization/presets/` (not `configs/`) so they ship with the wheel.
- Venv uses `--system-site-packages` because PyPI's file host (files.pythonhosted.org) was unreachable; matching torchvision/torchaudio were installed from the PyTorch CPU index, `nltk` from its GitHub source zip (explicitly approved by the owner), and a stub `spacy` in the venv blocks the system spacy (built for numpy 1.x, crashes pyannote's optional import).
- The original `pyannote/speaker-diarization` (2.x config) loads and runs on pyannote.audio 3.3.2, with version warnings from the checkpoints. No fallback to 3.1 was needed.

## Discrepancies (legacy code vs paper) and legacy quirks
See `AUDIT.md` D1-D8. Additions found by running things:
- **D9**: under the paper's verb lemmatization, lexicon entries `n***` and `s***s` can never match (`s***s` -> `s***`), in addition to the multi-word entry `son of a b****`. Kept in `paper`.
- **D10**: lexicon B has no `piss` (replaced by a different term vs lexicon A), so "piss" is missed by `paper`.
- **D11**: the `SPEAKER_01 -> FEMALE` heuristic is order/label based, not gender: on the synthetic set the **male** TTS voice was assigned `SPEAKER_01` and reported as FEMALE. The paper's gender-level claims are not supported by this mechanism.
  - Opt-in fix added: `speaker_labels: pitch_gender` (in `improved`) classifies each *turn* by median F0 (threshold 165 Hz, `UNKNOWN` when too little voiced audio). Measured: **25/25** correct on the synthetic clip (paper rule: 0/25). **Not validated on real audio.** On the Netflix trailer 6/15 turns sit at 156-172 Hz (ambiguous) and one at 400 Hz (search limit); on the New Yorker clip (clean 2-speaker diarization) the paper rule is coincidentally right and per-turn pitch flips some of the father's turns to FEMALE (FEMALE 9 vs 4), so per-turn pitch can be *worse* than the label rule when diarization is clean. A speaker-smoothed variant exists (`gender.smooth_by_speaker`, not wired into the pipeline) pending hand labels: fill `eval/data/gender_template.csv` and run `python -m eval.gender_eval score`.
  - Diarization itself merges several real speakers into one label on the trailer (3 labels), which no gender rule can undo at speaker level.
- **D13 (diarizer comparison)**: `pyannote/speaker-diarization-3.1` (preset `improved31`) vs the original on the same inputs. Netflix trailer: 30 turns vs 15, diarize 153 s vs 360 s, ASR 198 s vs 70 s, profane words reaching the transcript 10/13 vs 13/13 (short turns lose words); both give 3 speaker labels. Synthetic: both 2/2 speakers, purity 100%; the paper gender rule scores 0% (original) vs 100% (3.1) on identical audio, pitch gender 100% for both. New Yorker clip with 3.1: 120 turns vs 69 (52 vs 19 under 1 s), diarize 413 s + ASR 435 s, 31/32 gold words in the transcript, token F1 88.5% (paper) / 95.4% (improved) vs 88.9% / 95.5% with the original diarizer; speakers separated correctly (daughter vs father). Verdict: no clear win for 3.1 with per-turn transcription.
- **D12**: overlapping diarization turns are transcribed twice (seen on clip 1 at ~66-68 s). Not fixed (opt-in work pending).
- Whisper can hallucinate on silence/music (clip 2 segment 47 is garbled text).

## Evaluation caveats (be honest in any write-up)
- Gold labels for the two real clips were assigned by the model, not an independent human; they need an audit. Borderline calls are listed in the label files' notes (e.g. `h***`, `screwed over`).
- `improved` was designed from clip 1's errors; clip 2 and the synthetic set are out-of-sample for it, but the author of both is the same model/session.
- The paper's reported 94.8% accuracy / 93.4% F1 cannot be reproduced from this folder (no code or labels); our numbers measure a different thing (token-level detection on 2 clips) and are not comparable.
- Labeled transcripts live in `eval/data/` (gitignored: copyrighted dialogue); only masked aggregates are committed.

## Blockers / for the owner
- Revoke the old leaked HF token (the current `.env` token is separate; keep it out of git).
- Audit the labels in `eval/data/*.labels.json`.
- Memory: the machine was low on RAM; long runs were killed twice by the harness. Close browser tabs before model runs.
