import whisper
from pyannote.audio import Pipeline
from pydub import AudioSegment
import tempfile
import os
import string
import subprocess
import nltk
import torch
from nltk.stem import WordNetLemmatizer

# --- Setup for lemmatization ---
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# --- Step 1: Load models ---
# Load Whisper model (uses GPU automatically if available)
whisper_model = whisper.load_model("small")

# Load PyAnnote speaker diarization pipeline
print("Loading speaker diarization pipeline...")
diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")

# Move PyAnnote pipeline to GPU
diarization_pipeline.to(torch.device("cuda"))

# --- Step 2: Define cuss words ---
cuss_words = {
    "damn", "hell", "shit", "fuck", "bitch", "bastard", "ass", "crap", "dick", "pussy",
    "son of a bitch", "jerk", "freak", "screw", "moron", "idiot", "dumb", "nuts", "sucks", "heck"
}

# --- Step 3: Extract audio from .mp4 video using ffmpeg ---
original_video_path = "shortmovie1.mp4"
temp_audio_path = "temp_audio.wav"

print("Extracting audio from video using ffmpeg...")
audio = AudioSegment.from_file(original_video_path, format="mp4")
audio = audio.set_channels(1).set_frame_rate(16000)
audio.export(temp_audio_path, format="wav")

# --- Step 4: Speaker diarization ---
print("Performing speaker diarization (this takes a moment)...")
diarization = diarization_pipeline(temp_audio_path)

# Reload the audio for segment slicing
audio = AudioSegment.from_wav(temp_audio_path)

# --- Step 5: Speaker mapping and processing ---
speaker_map = {}
speaker_counter = {"MALE": 0, "FEMALE": 0}
gender_transcripts = {"MALE": "", "FEMALE": ""}
gender_cuss_count = {"MALE": 0, "FEMALE": 0}
gender_cuss_words = {"MALE": [], "FEMALE": []}
conversation = []

for turn, _, speaker in diarization.itertracks(yield_label=True):
    start_ms = int(turn.start * 1000)
    end_ms = int(turn.end * 1000)

    segment = audio[start_ms:end_ms]

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
        segment.export(temp_audio_file.name, format="wav")
        temp_path = temp_audio_file.name

    result = whisper_model.transcribe(temp_path, language="en")
    text = result["text"].strip()

    # Simple gender assignment heuristic
    if speaker not in speaker_map:
        if "SPEAKER_01" in speaker and speaker_counter["FEMALE"] == 0:
            speaker_map[speaker] = "FEMALE"
            speaker_counter["FEMALE"] += 1
        else:
            speaker_map[speaker] = "MALE"
            speaker_counter["MALE"] += 1

    speaker_gender = speaker_map[speaker]

    if text:
        conversation.append(f"{speaker_gender}: {text}")
        gender_transcripts[speaker_gender] += " " + text

    os.remove(temp_path)

# --- Step 6: Clean up temp audio ---
os.remove(temp_audio_path)

# --- Step 7: Analyze cuss words per gender with lemmatization ---
for gender, transcript in gender_transcripts.items():
    words = transcript.lower().split()
    cleaned_words = []
    for word in words:
        word = word.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
        lemma = lemmatizer.lemmatize(word, pos="v")  # Verb lemmatization (e.g., "fucked" -> "fuck")
        cleaned_words.append(lemma)

    used = [word for word in cleaned_words if word in cuss_words]
    gender_cuss_words[gender] = used
    gender_cuss_count[gender] = len(used)

# --- Step 8: Output results ---
print("\n--- FULL TRANSCRIPTION ---\n")
for line in conversation:
    print(line)

print("\n--- CUSS WORD ANALYSIS ---")
for gender in ["MALE", "FEMALE"]:
    print(f"\n🔹 {gender}")
    print(f"Transcript: {gender_transcripts[gender].strip()}")
    unique_cuss = set(gender_cuss_words[gender])
    for word in unique_cuss:
        print(f"- {word} (count: {gender_cuss_words[gender].count(word)})")
    print(f"Total cuss words: {gender_cuss_count[gender]}")
    print(f"Unique cuss words: {len(unique_cuss)}")
