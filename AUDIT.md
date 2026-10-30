# Legacy code audit

Scope: the author's original project folder (`LEGACY_DIR`), read-only. The new repo was created in a subfolder, `cuss-word-pipeline/`, which is excluded from this audit.
Sources read: all `.py`/`.txt` files in full, the published paper PDF in full, the BlackBook report (keyword scan only), and the sample-output PDF header. Profane terms are masked in this document.

> **SECURITY: a Hugging Face access token (`hf_…`, 37 chars) is hard-coded in `Code/final.py` (line 31) and also sits alone in `Code/tempCodeRunnerFile.py`. Revoke it at https://huggingface.co/settings/tokens. The value is not reproduced anywhere in this repo.**

## A.1 Inventory

| Path | Type | Size | What it does | Stage | Recommendation |
|---|---|---|---|---|---|
| `Code/final.py` | py | 3.9 KB | v1 CLI: ffmpeg-python audio extraction → pyannote diarization → per-turn Whisper `small` → exact-match lexicon A. **Contains the HF token.** | all | `drop` (superseded; useful as an exact-match / no-lemma reference for the `legacy_v1` preset) |
| `Code/final_1.py` | py | 3.3 KB | v1b: same as `final.py` but reads a `.wav`, no token, no ffmpeg step. | all | `drop` (superseded) |
| `Code/tempCodeRunnerFile.py` | py | 37 B | Editor scratch file holding only the HF token. | none | `drop` (secret) |
| `Code/requirements.txt` | txt | 61 B | `whisper` (wrong PyPI package), `tempfile` (stdlib, not installable), unpinned torch/pyannote/pydub. | env | `rewrite` |
| `Code/sample-video.mp4` | mp4 | 6.4 MB | Byte-identical to the Netflix trailer at the top level. Copyrighted. | data | `drop` (not committable) |
| `Code/sample-audio.wav` | wav | 3.8 MB | Audio of the above. Copyrighted derivative. | data | `drop` |
| `Code/temp_audio.wav` | wav | 3.8 MB | Byte-identical copy of `sample-audio.wav`; a leftover from a crashed run. | data | `drop` |
| `Code/sampleaudio3.wav` | wav | 12.5 MB | Unknown audio, source and license unknown. | data | `unsure` (default: drop) |
| `Code/New Microsoft Word Document.pdf` | pdf | 4.0 MB | Pasted console outputs ("Video 1 Output", …) with unmasked transcripts of copyrighted clips. | docs | `drop` (reference only; cannot publish) |
| `code-new/final_gpu.py` | py | 4.2 KB | **Canonical CLI** (named in paper §3.2). mp4 → pydub → pyannote on CUDA → per-turn Whisper `small` → lemmatized (verb) match, lexicon B. Input `shortmovie1.mp4` is missing from the folder. | all | `keep` → port into `paper` preset |
| `code-new/app.py` | py | 5.4 KB | **Canonical GUI** (paper §3.2). Streamlit upload; identical logic to `final_gpu.py` plus bar and pie charts. Everything runs at import time. | all + viz | `merge into` the pipeline; GUI is not in scope for v0.1.0 (optional later) |
| `code-new/requirements.txt` | txt | 97 B | Unpinned: streamlit, openai-whisper, pyannote.audio, torch, pydub, nltk, matplotlib, seaborn, ffmpeg-python. | env | `rewrite` (pin from real runs) |
| `code-new/venv/**` | dir | ~22k files | Windows venv, Python 3.9.13. Has only numpy 2.0.2, openai-whisper 20240930 and torch 2.7.0 as dist-infos. pyannote, nltk, streamlit and pydub are **not** installed in it. | env | `drop` |
| `project/code/{app.py,final_gpu.py,requirements.txt}` | py/txt | – | SHA-256 identical to the `code-new/` copies. | – | `drop` (duplicate) |
| `project/code/venv/**` | dir | ~22k files | Copy of the same venv. | env | `drop` |
| `project/Report_Identification of Cuss Words.pdf` | pdf | 847 KB | Identical to `BlackBookReport_…pdf`. | docs | `drop` (duplicate) |
| `BlackBookReport_Identification of Cuss Words.pdf` | pdf | 847 KB | Project report (1932 text lines). Used for context only. | docs | `keep` outside the repo (cite only) |
| `project/ResearchPaper_Identification of Cuss Words.docx` | docx | 717 KB | Draft of the paper. | docs | `drop` (the published PDF is canonical) |
| `Published_ResearchPaper_Identification of Cuss Words.pdf` | pdf | 250 KB | Published IJFMR paper (Vol 7, Issue 3, May–June 2025, ID IJFMR250344377). | docs | `keep` outside the repo (copyright of the journal); cite via DOI |
| `project/implementations/Screenshot *.jpg` (6) | jpg | 108–259 KB | Streamlit GUI screenshots (paper Figs 1–5). Not opened. They may show unmasked terms or copyrighted frames. | docs | `unsure` (default: drop, re-capture masked ones on a CC clip) |
| `History of Swear Words - Official Trailer - Netflix.mp4` | mp4 | 6.4 MB | Copyrighted trailer; same bytes as `Code/sample-video.mp4`. | data | `drop` |
| `A Father-Daughter Swearing Lesson - “The F-Word” - The New Yorker.mp4` | mp4 | 19 MB | Copyrighted video. | data | `drop` |
| `CLAUDE_CODE_PROMPT_speech_pipeline.md` | md | 18 KB | The brief for this task. | – | not part of the repo |

### Duplicate groups and the canonical pick
- **Pipeline script**: `final.py` (21:07) → `final_1.py` (23:08), both from 2025-04-18 by mtime. Then `final_gpu.py` and `app.py`, which add lemmatization and CUDA. The folder has no timestamps for those, so I picked the canonical files **by content, not date**: they are the only ones the paper names (§3.2) and the only ones with lemmatization (§3.6). The v1 scripts are exact-match only.
- **`code-new/` ≡ `project/code/`**: byte-identical (SHA-256 matched for all three files).
- **Report PDF ×2**, **sample audio ×2**, **sample video ≡ Netflix mp4**: byte-identical.

## A.2 Reconstructed pipeline (as the code actually runs)

Note: this is **not** "ASR → word timestamps → align to speakers". The legacy flow is diarize first, then transcribe each turn separately.

1. **Input / audio extraction** (`final_gpu.py:37-43`): `AudioSegment.from_file(video, format="mp4")` → `set_channels(1).set_frame_rate(16000)` → export `temp_audio.wav` in the cwd. (v1 `final.py:15-21` used ffmpeg-python with `pcm_s16le`/16 kHz/mono; the paper §3.3 shows the pydub form.)
2. **Diarization** (`final_gpu.py:25-28,47`): `Pipeline.from_pretrained("pyannote/speaker-diarization")` with **no token argument**, moved to `cuda`, default parameters. pyannote.audio version is not pinned or recorded anywhere.
3. **Per-turn ASR** (`final_gpu.py:21,60-71`): for each `itertracks` turn, slice the audio by `int(start*1000)`..`int(end*1000)` ms, export to a temp wav, run `whisper.load_model("small").transcribe(path, language="en")` with otherwise default decoding (no word timestamps, no VAD, temperature fallback defaults). Only `result["text"].strip()` is kept. **Timestamps are discarded.**
4. **Speaker → "gender"** (`:74-80`): the first time a label is seen, if `"SPEAKER_01" in label` and no FEMALE yet → FEMALE, otherwise MALE. This is a label heuristic, not gender detection.
5. **Transcript assembly** (`:84-86`): non-empty turn text is appended as `"{GENDER}: {text}"` and concatenated per gender.
6. **Profanity detection** (`:94-104`): per gender, `transcript.lower().split()` (whitespace) → remove **all** punctuation chars inside the token → `WordNetLemmatizer().lemmatize(w, pos="v")` → exact set membership in lexicon B (20 entries). Counts are per token (repeats counted).
7. **Outputs**: console prints (`final_gpu.py`) or Streamlit text plus bar/pie charts (`app.py`). **No files are written, no timestamps, no per-speaker hits with positions.**

Lexicon A (v1, `final.py:51-54`) and lexicon B (`final_gpu.py:31-34`, `app.py:28-31`) are 20 entries each; they differ in exactly one entry (A has a "p***" term where B has "p****"). Entries include one multi-word phrase and mild words like "heck", "freak", "idiot", "dumb", "nuts".

Language: forced `en`. No Hindi/Hinglish handling anywhere.

## A.3 Environment (evidence-based)

- Python 3.9.13 (`venv/pyvenv.cfg`; paper/report say "Python 3.9+").
- Installed in the venv: `openai-whisper 20240930`, `torch 2.7.0`, `numpy 2.0.2`. **Nothing else of the stack** (pyannote, nltk, streamlit, pydub). So I cannot recover the pyannote/nltk versions; they are unpinned in `requirements.txt`.
- Imports used: `whisper`, `pyannote.audio.Pipeline`, `pydub`, `nltk` (WordNet), `torch`, `streamlit`, `matplotlib` (+ unused `seaborn`, `subprocess`), `ffmpeg-python` (v1 only).
- Hardware assumption: CUDA GPU, hard-coded `torch.device("cuda")`, TF32 flags. The development machine has a GTX 1650 Ti (4 GB); the installed torch there was CPU-only. The report lists "CUDA-enabled PyTorch" and 100 GB storage.
- Hard-coded paths: relative `shortmovie1.mp4`, `sample-video.mp4`, `sample-audio.wav`, and `temp_audio.wav` in the cwd (collides when two runs share a folder). No absolute personal paths found in code.
- Network at runtime: `nltk.download('wordnet')` at every import, and model downloads on first use (pyannote is gated and needs an HF login).

## A.4 Quality findings (ranked by impact)

1. **Leaked HF token** in two files (see the top banner).
2. **Paper vs. code gaps** (details in "Discrepancies" below): no code or labeled data in the folder behind the reported 94.8% / 93.4% F1; no evaluation script at all.
3. **Overlapping turns are transcribed twice.** pyannote can emit overlapping turns; each is sliced and transcribed separately, so the same speech can be counted for two speakers. The paper claims robustness to overlap.
4. **Per-turn transcription can cut words at boundaries** (sample output PDF shows a truncated "fo" glued to the next line) and loses Whisper's cross-turn context.
5. **The multi-word lexicon entry can never match**: tokens are split on whitespace before lookup, so a phrase entry is dead.
6. **`requirements.txt` is broken**: `whisper` on PyPI is a different package from `openai-whisper`, and `tempfile` is stdlib. Nothing is pinned.
7. **Not offline**: `nltk.download` each run; first-run model downloads; the gated pyannote model needs a token that the canonical scripts never pass (they only work if the user ran `huggingface-cli login`).
8. **No GPU fallback**: `.to(cuda)` crashes on CPU-only machines.
9. **No error handling or cleanup guarantees**: temp files are removed only on the happy path (leaked `temp_audio.wav` and `sample-audio.wav` are evidence); no `try/finally`.
10. **Copy-paste**: the whole pipeline is duplicated in 4 scripts, and `app.py` re-implements `final_gpu.py` and runs everything at import/module level.
11. **Dead code**: unused imports (`subprocess`, `seaborn`), `gender_cuss_count` pre-initialisation overwritten, `speaker_counter["MALE"]` never read.
12. Copyrighted media sitting next to the code; stray venv copies (~44k files).

## Discrepancies: paper vs. code (to be recorded in `docs/PROGRESS.md`)

| # | Paper says | Code does | Handling |
|---|---|---|---|
| D1 | "timestamped, speaker-labelled" output (BlackBook); my brief assumes word-level timestamps and alignment | Segment text only; no timestamps kept | `paper` preset: reproduce segment-level flow; hits carry the **turn** start/end (data the code had but threw away). Word-level alignment is opt-in. |
| D2 | Metrics: accuracy 94.8%, precision 91.2%, recall 95.7%, F1 93.4% | No evaluation code, data or labels in the folder | Cannot reproduce. Docs will say "reported in the paper, not reproduced here". Phase E measures on public data. |
| D3 | "PyAnnote was robust to overlapping speech" | Overlap is not handled (double transcription) | Keep in `paper`; opt-in overlap handling. |
| D4 | Speaker-gender mapping | Label heuristic: `SPEAKER_01` → FEMALE, else MALE; not gender detection | Keep in `paper` but name it "label heuristic" in docs; opt-in neutral speaker labels. |
| D5 | "Offline" | Needs network for nltk + model download + HF token | Honest "offline after one-time download" section; `prefetch-models`. |
| D6 | "Lemmatization-based filter" with `"fuck", "shit", "bitch", "damn", ...` | Lexicon B, with `pos="v"` lemmatization only | Port exactly. Verb-only lemmatization is a limitation (noun plurals may not normalise). |
| D7 | Lexicon in paper vs. code | v1 scripts (lexicon A) differ from GUI/GPU (lexicon B) by one term | Use B as `paper`; offer A as `legacy_v1`. |
| D8 | "Evaluation on Inception (2010) / Tenet (2020)", "Tenet ~50% higher count" | Copyrighted films; no data or outputs | Not reproducible; not committed. |

## A.5 Questions for you (defaults in brackets)

1. **Canonical files**: confirm `code-new/final_gpu.py` + `code-new/app.py` (lexicon B, lemmatized) define the `paper` preset, with the v1 scripts only as an optional `legacy_v1` preset. [Yes]
2. **Package/repo name**: the folder is `cuss-word-pipeline`, which sits *inside* the legacy folder. Python package name and GitHub repo name? [package `profanity_diarization`, repo `profanity-diarization-pipeline`]
3. **Token**: have you revoked the leaked HF token? (I will not use it.) Do you have a fresh token + accepted terms for `pyannote/speaker-diarization` (and its dependencies `pyannote/segmentation`)? [Revoke now; supply `HF_TOKEN` via env]
4. **pyannote version**: the paper's `pyannote/speaker-diarization` is the legacy 2.x-style pipeline. Should `paper` pin `pyannote.audio` 2.1.x / 3.x, whichever runs on your machine, or the version you originally used if you remember it? [Whichever installs and runs; I'll record the exact version]
5. **Lexicon provenance and license**: did you write the 20 words yourself or take them from a list? [Own list, MIT; confirm]
6. **Fidelity check clip**: which CC0/CC-BY clip with 2+ speakers can I use for `compare_with_legacy.py`? [A LibriVox / AMI excerpt, or a TTS-generated dialogue]
7. **Screenshots / `sampleaudio3.wav`**: are these publishable? [No; drop and re-capture later with masked terms]
8. **Hinglish/Hindi**: the code has none, though the paper's future work mentions multilingual. [Do not claim it; add only a documented hook in D.5]
9. **GPU**: the development machine has a GTX 1650 Ti (4 GB). Is running the real models for the fidelity check and eval here OK? [Yes; Whisper `small` + pyannote fit, but slowly]
10. **GUI**: keep a Streamlit GUI (`app.py`) in the new repo as an optional extra? [No for v0.1.0, maybe later]
11. **Licence and authorship**: MIT, author "Rusheil Singh Baath" (co-authors in CITATION.cff as the paper lists). Is this right? [Yes]
12. **Nesting**: the repo stays inside the legacy folder as you asked. A later move or clone to a standalone location is trivial; fine? [Yes]
