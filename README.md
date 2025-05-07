# Cuss Word Detection and Speaker Diarization

This project identifies offensive language from video/audio content and attributes it to individual speakers using **Whisper** (ASR) and **PyAnnote** (speaker diarization). It provides clear breakdowns of who said what, with profanity statistics and visual comparisons across speakers.

## 🎓 Project Overview

Developed as part of the final-year capstone at **MIT World Peace University, Pune**, this tool helps analyze dialogues in multimedia content for:
- Content moderation
- Annotated transcripts
- Educational audits
- Social media analysis

## 🧠 Key Features

- 🎙️ **Speaker Diarization**: Segments speech and identifies speaker turns
- 🗣️ **Speech Transcription**: Converts audio to text using Whisper
- 🤬 **Cuss Word Detection**: Flags and counts offensive words
- 📊 **Visualization**: Bar and pie charts for profanity use by gender
- 🖥️ **Streamlit Interface**: Interactive UI for uploads, transcripts, and analysis

## 🚀 Tech Stack

| Tool            | Purpose                                |
|-----------------|----------------------------------------|
| `Whisper`       | Speech-to-text transcription (OpenAI) |
| `PyAnnote`      | Speaker diarization (pretrained model) |
| `Pydub`         | Audio slicing and format handling      |
| `NLTK`          | Lemmatization and profanity detection  |
| `Streamlit`     | Interactive web interface              |
| `Matplotlib`, `Seaborn` | Visualization and plotting     |
| `Torch`         | GPU acceleration for inference         |
