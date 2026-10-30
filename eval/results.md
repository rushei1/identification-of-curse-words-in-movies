# Evaluation results

Unit: **token** (whitespace-split ASR output); *profane* is the positive class. Accuracy is dominated by true negatives (most words are not profane), so read precision, recall and F1 first. All profane terms are masked.

Run: 2026-10-30T20:36:21.597931+00:00 | Python 3.9.13 | Windows-10-10.0.26300-SP0

## Headline: real clips pooled (sample-video + newyorker)

1121 tokens, 45 gold-positive (micro-average: confusion matrices summed, then rescored). Labels are model-assigned, not independent human annotation; the `improved` matcher was designed from the first clip, so only the second clip and the synthetic set are out-of-sample for it.

| Preset | TP | FP | FN | TN | Accuracy | Precision | Recall | F1 | Specificity | FPR | FNR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| paper | 38 | 3 | 7 | 1073 | 99.1% | 92.7% | 84.4% | 88.4% | 99.7% | 0.3% | 15.6% |
| improved | 45 | 3 | 0 | 1073 | 99.7% | 93.8% | 100.0% | 96.8% | 99.7% | 0.3% | 0.0% |

## netflix_d31

273 tokens, 10 gold-positive. Labels by: derived: a token is positive iff its normalized form is a gold form of the hand-labeled run with the original diarizer. Netflix trailer re-transcribed with the pyannote 3.1 diarizer (different turn boundaries, so index labels do not apply). Gold token count here vs 13 in the original run shows how many profane words ASR still produced.

| Preset | TP | FP | FN | TN | Accuracy | Precision | Recall | F1 | Specificity | FPR | FNR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| paper | 7 | 0 | 3 | 263 | 98.9% | 100.0% | 70.0% | 82.4% | 100.0% | 0.0% | 30.0% |
| improved | 10 | 0 | 0 | 263 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% |

Stage time (CPU): audio 0s, diarize 153s, asr 198s

- **paper** false positives: none; false negatives: {'h***': 1, 'c***': 1, 'M***': 1}
- **improved** false positives: none; false negatives: none

## newyorker

847 tokens, 32 gold-positive. Labels by: Claude (model-assigned, NOT independent human annotation; please audit). Real clip 2 (held-out: the improved matcher was designed from clip 1 only). Policy: positive = vulgar/expletive/swear word or personal-insult noun, used or mentioned, inflected/compound forms included. Negative = mild evaluative words (*** s*** the '*** half of a compound, the idiom 'screwed over', the minced oath '*** (BORDERLINE: overrule if you disagree). Segment 47 is a Whisper hallucination (garbled text) and contains no gold positives.

| Preset | TP | FP | FN | TN | Accuracy | Precision | Recall | F1 | Specificity | FPR | FNR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| paper | 28 | 3 | 4 | 812 | 99.2% | 90.3% | 87.5% | 88.9% | 99.6% | 0.4% | 12.5% |
| improved | 32 | 3 | 0 | 812 | 99.6% | 91.4% | 100.0% | 95.5% | 99.6% | 0.4% | 0.0% |

Stage time: not recorded (report rebuilt from the stage cache).

- **paper** false positives: {'d***': 1, 's***': 1, 'h***': 1}; false negatives: {'f***': 3, 'M***': 1}
- **improved** false positives: {'d***': 1, 's***': 1, 'h***': 1}; false negatives: none

## newyorker_d31

822 tokens, 31 gold-positive. Labels by: derived: a token is positive iff its normalized form is a gold form of the hand-labeled run with the original diarizer. Real clip 2 re-transcribed with the pyannote 3.1 diarizer (different turn boundaries, so index labels do not apply).

| Preset | TP | FP | FN | TN | Accuracy | Precision | Recall | F1 | Specificity | FPR | FNR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| paper | 27 | 3 | 4 | 788 | 99.1% | 90.0% | 87.1% | 88.5% | 99.6% | 0.4% | 12.9% |
| improved | 31 | 3 | 0 | 788 | 99.6% | 91.2% | 100.0% | 95.4% | 99.6% | 0.4% | 0.0% |

Stage time (CPU): audio 0s, diarize 413s, asr 435s

- **paper** false positives: {'d***': 1, 's***': 1, 'h***': 1}; false negatives: {'f***': 3, 'm***': 1}
- **improved** false positives: {'d***': 1, 's***': 1, 'h***': 1}; false negatives: none

## sample-video

274 tokens, 13 gold-positive. Labels by: Claude (model-assigned, NOT independent human annotation; please audit). Policy: a token is positive if it is a profane word, used or mentioned (inflected/compound forms included). Literal homographs of mild words would be negative (none occur here). Real clip 1; the improved matcher was designed after seeing this clip's errors, so its numbers here are in-sample.

| Preset | TP | FP | FN | TN | Accuracy | Precision | Recall | F1 | Specificity | FPR | FNR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| paper | 10 | 0 | 3 | 261 | 98.9% | 100.0% | 76.9% | 87.0% | 100.0% | 0.0% | 23.1% |
| improved | 13 | 0 | 0 | 261 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% |

Stage time (CPU): audio 0s, diarize 360s, asr 70s

- **paper** false positives: none; false negatives: {'h***': 1, 'c***': 1, 'M***': 1}
- **improved** false positives: none; false negatives: none

## synthetic

205 tokens, 15 gold-positive. Labels by: by construction (scripted TTS text; known positives). SYNTHETIC, not representative of real movies. 15 scripted profane words; 15 recognised verbatim by ASR in the transcript, 0 of 14 distinct forms never appeared (ASR miss, not a detector miss). Policy as for the real clips: mild adjectives/verbs (*** '*** and literal uses ('screwed', '*** '*** are negative; '*** as an insult noun is positive. 'Moby D*** (a name) is a known ambiguity for lexical matchers. 'Dickens' exercised the stoplist; TTS respelled S*** so those two are covered only by unit tests.

| Preset | TP | FP | FN | TN | Accuracy | Precision | Recall | F1 | Specificity | FPR | FNR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| paper | 10 | 4 | 5 | 186 | 95.6% | 71.4% | 66.7% | 69.0% | 97.9% | 2.1% | 33.3% |
| improved | 15 | 4 | 0 | 186 | 98.0% | 78.9% | 100.0% | 88.2% | 97.9% | 2.1% | 0.0% |

Stage time (CPU): audio 0s, diarize 247s, asr 107s

- **paper** false positives: {'s***': 1, 'f***': 1, 'D***': 1, 'd***': 1}; false negatives: {'b***': 1, 'm***': 1, 'a***': 1, 'd***': 1, 'p***': 1}
- **improved** false positives: {'s***': 1, 'f***': 1, 'D***': 1, 'd***': 1}; false negatives: none

## Reading the errors

- **False negatives** are words outside the 20-term lexicon or in a form it cannot match (compounds, agent nouns such as the *-er* form of a verb). Exact-match `paper` cannot catch them.
- **False positives** are lexicon words used in a mild or literal sense (`screwed`, `*** `*** `*** and names (a surname or a book title). Neither preset uses context, so `improved` does not reduce them.
- Gold labels cover **detection on the ASR output**. Words Whisper mis-hears or drops are invisible to this metric; the synthetic set reports how many scripted words ASR recognised.
- Token-level counts differ from the paper's hit counting only for multi-word phrases.

## Diarizer comparison on the synthetic clip (truth known)

| Diarizer | Speakers found / true | Turns | Purity | Gender: paper rule | Gender: pitch per turn |
|---|---|---|---|---|---|
| pyannote/speaker-diarization (paper) | 2 / 2 | 25 | 100.0% | 0.0% | 100.0% |
| pyannote/speaker-diarization-3.1 | 2 / 2 | 27 | 100.0% | 100.0% | 100.0% |

## Not run

Nothing skipped.
