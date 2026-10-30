# Changelog

## [Unreleased]

### Added
- Package `profanity_diarization`: ffmpeg audio extraction, pyannote diarization, per-turn Whisper transcription,
  lexicon matching, JSON/CSV/SRT/Markdown reports (terms masked in SRT/Markdown), per-stage caching, Typer CLI
  (`process`, `batch`, `prefetch-models`, `report`).
- Presets: `paper` (faithful to the published code), `legacy_v1`, and opt-in `improved` / `improved31`.
- `improved` matching: stem matching with an innocent-word stoplist and whole-phrase matching.
- `pitch_gender`: per-turn voice-pitch gender with an audit table (unvalidated on real audio).
- Evaluation tooling in `eval/` (token-level metrics, diarizer comparison on synthetic data, masked report),
  a figures script, and documentation of data and caveats.
- Model-free tests, ruff configuration, GitHub Actions CI, Dockerfile.

### Known issues (kept in `paper` on purpose)
Turn-level timestamps only, overlapping turns transcribed twice, unreachable lexicon entries, label-based
"gender". See `docs/PROGRESS.md`.
