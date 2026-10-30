# Progress

Legacy dir: `F:\Projects\identification_of_cuss_words` (read-only). Repo: this folder, nested inside it by the owner's choice.

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
- **D9**: under the paper's verb lemmatization, lexicon entries `nuts` and `sucks` can never match (`sucks` -> `suck`), in addition to the multi-word entry `son of a b****`. Kept in `paper`.
- **D10**: lexicon B has no `piss` (replaced by a different term vs lexicon A), so "piss" is missed by `paper`.
- **D11**: the `SPEAKER_01 -> FEMALE` heuristic is order/label based, not gender: on the synthetic set the **male** TTS voice was assigned `SPEAKER_01` and reported as FEMALE. The paper's gender-level claims are not supported by this mechanism.
- **D12**: overlapping diarization turns are transcribed twice (seen on clip 1 at ~66-68 s). Not fixed (opt-in work pending).
- Whisper can hallucinate on silence/music (clip 2 segment 47 is garbled text).

## Evaluation caveats (be honest in any write-up)
- Gold labels for the two real clips were assigned by the model, not an independent human; they need an audit. Borderline calls are listed in the label files' notes (e.g. `heck`, `screwed over`).
- `improved` was designed from clip 1's errors; clip 2 and the synthetic set are out-of-sample for it, but the author of both is the same model/session.
- The paper's reported 94.8% accuracy / 93.4% F1 cannot be reproduced from this folder (no code or labels); our numbers measure a different thing (token-level detection on 2 clips) and are not comparable.
- Labeled transcripts live in `eval/data/` (gitignored: copyrighted dialogue); only masked aggregates are committed.

## Blockers / for the owner
- Revoke the old leaked HF token (the current `.env` token is separate; keep it out of git).
- Audit the labels in `eval/data/*.labels.json`.
- Memory: the machine was low on RAM; long runs were killed twice by the harness. Close browser tabs before model runs.
