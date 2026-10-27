# Progress

Legacy dir: `F:\Projects\identification_of_cuss_words` (read-only). Repo: this folder, nested inside it by the owner's choice.

## Status
- [x] A.1 Inventory (`AUDIT.md`)
- [x] A.2 Pipeline reconstruction
- [x] A.3 Environment
- [x] A.4 Quality findings
- [x] A.5 Questions → **Phase A gate: waiting for answers**
- [ ] Phase B ...

## Decisions
- LEGACY_DIR = parent folder of this repo; repo created as a subfolder (owner's instruction).
- Brief assumes word-level ASR + alignment; legacy is diarize-then-transcribe-per-turn. Trusting the code (brief rule 9): `paper` preset follows the legacy flow.

## Discrepancies
See `AUDIT.md` → "Discrepancies" D1–D8.

## Blockers
- Leaked HF token in legacy code: owner must revoke it.
