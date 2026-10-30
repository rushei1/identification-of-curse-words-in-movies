# Data used and why it is not in this repository

**No audio or video is committed.** The test clips are copyrighted, and generated audio is not bundled.
This file records exactly what was used so the results can be reproduced.

## Test videos (not included)

| Name | Source | Length | Size | SHA-256 |
|---|---|---|---|---|
| "History of Swear Words - Official Trailer - Netflix.mp4" | Official Netflix trailer, downloaded from the public web | 120 s | 6,409,669 B | `69159c66e68600c7930b5fcf18a92dda260453d1b3695b15492a7aa83b760cb0` |
| "A Father-Daughter Swearing Lesson - The F-Word - The New Yorker.mp4" | The New Yorker's public video | 392 s | 19,203,916 B | `77bba27cca57604318fc79a41c4b6abafa3d0418f77bced9730975e057b29daf` |

Get your own copy from the original publisher. If your file's SHA-256 matches the one above, it is the same file and
should reproduce the reported transcripts up to Whisper's decoding variation. Verify with `sha256sum file.mp4`
(Linux/macOS) or `Get-FileHash file.mp4` (PowerShell). Otherwise your numbers may differ.

Real-clip numbers in the README were computed on these two files. The pipeline caches by file *content*, so a
renamed copy is treated as the same input.

## Synthetic clip (generated, not included)

`python -m eval.synthetic audio` renders 25 scripted sentences with the two offline Windows text-to-speech voices
(*Microsoft David Desktop*, male, and *Microsoft Zira Desktop*, female) into `eval/data/synthetic.wav`. The script is
in `eval/synthetic.py`; words prefixed with `*` are the profane positives, so the labels are exact by
construction. Audio produced by a system voice may be subject to that voice's licence terms, so it is generated
locally instead of being committed. Requires Windows (`System.Speech`) and ffmpeg.

## Labels

| File (git-ignored, under `eval/data/`) | What it is |
|---|---|
| `sample-video.labels.json`, `newyorker.labels.json` | Token-level gold positives for the two real clips. **Assigned by the AI model that helped write this code, not by independent human annotators.** Please audit; borderline calls are listed in the `notes` field |
| `*_d31.labels.json` | The same gold *forms* applied to the transcripts produced with the 3.1 diarizer |
| `synthetic.labels.json` | By construction (scripted text) |
| `gender_template.csv` | Blank sheet to hand-label speaker gender per turn (see the README) |

They are git-ignored because they contain transcripts of copyrighted dialogue. Only the masked aggregate report
(`eval/results.md`) is committed; the raw `eval/results/*.json` files are git-ignored too.

## Lexicon

`src/profanity_diarization/lexicon/lexicon_b.txt` is the 20-term list from the paper's code (`lexicon_a.txt` is the
earlier list). The author wrote them for the project; they are released under the repository's MIT licence.
See `data/lexicon/README.md`.

## Models (downloaded at first use)

| Model | Licence / access |
|---|---|
| OpenAI Whisper `small` | MIT |
| `pyannote/speaker-diarization`, `pyannote/segmentation` | Gated on Hugging Face: accept the terms, then use a token |
| `pyannote/speaker-diarization-3.1`, `pyannote/segmentation-3.0` | Gated, same process |
| NLTK WordNet | WordNet licence (permissive) |
