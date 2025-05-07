import streamlit as st
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
import matplotlib.pyplot as plt
import seaborn as sns

# --- Setup for lemmatization ---
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# --- Load models ---
whisper_model = whisper.load_model("small")
diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
diarization_pipeline.to(torch.device("cuda"))

# Define cuss words
cuss_words = {
    "damn", "hell", "shit", "fuck", "bitch", "bastard", "ass", "crap", "dick", "pussy",
    "son of a bitch", "jerk", "freak", "screw", "moron", "idiot", "dumb", "nuts", "sucks", "heck"
}

# Streamlit App Layout
st.title("Video Speaker Diarization and Cuss Word Analysis")

# File uploader for video file
uploaded_file = st.file_uploader("Upload a video file (MP4)", type=["mp4"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as temp_video_file:
        temp_video_path = temp_video_file.name
        temp_video_file.write(uploaded_file.read())
    
    st.write("Processing video...")

    # Step 3: Extract audio from .mp4 video using ffmpeg
    temp_audio_path = "temp_audio.wav"
    audio = AudioSegment.from_file(temp_video_path, format="mp4")
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(temp_audio_path, format="wav")
    
    # Step 4: Perform speaker diarization
    diarization = diarization_pipeline(temp_audio_path)
    audio = AudioSegment.from_wav(temp_audio_path)

    # Step 5: Process each speaker's speech and transcribe
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

        # Assign gender to speakers
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

    os.remove(temp_audio_path)

    # Step 7: Analyze cuss words
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

    # Step 8: Display results in Streamlit

    # Full transcription
    st.subheader("Full Transcription")
    for line in conversation:
        st.write(line)

    # Cuss word analysis
    st.subheader("Cuss Word Analysis")
    for gender in ["MALE", "FEMALE"]:
        st.write(f"🔹 {gender}")
        st.write(f"Transcript: {gender_transcripts[gender].strip()}")
        unique_cuss = set(gender_cuss_words[gender])
        for word in unique_cuss:
            st.write(f"- {word} (count: {gender_cuss_words[gender].count(word)})")
        st.write(f"Total cuss words: {gender_cuss_count[gender]}")
        st.write(f"Unique cuss words: {len(unique_cuss)}")

    # --- Plotting section ---
    st.subheader("Cuss Word Comparison Between Male and Female")

    # Plot 1: Bar chart comparing male and female cuss words
    fig, ax = plt.subplots()
    ax.bar(gender_cuss_count.keys(), gender_cuss_count.values(), color=["blue", "orange"])
    ax.set_title("Cuss Words Count by Gender")
    ax.set_ylabel("Cuss Words Count")
    ax.set_xlabel("Gender")
    st.pyplot(fig)

    # Plot 2: Pie chart of cuss word distribution (male vs female)
    fig, ax = plt.subplots()
    cuss_word_distribution = [gender_cuss_count["MALE"], gender_cuss_count["FEMALE"]]
    ax.pie(cuss_word_distribution, labels=["Male", "Female"], autopct='%1.1f%%', colors=["blue", "orange"])
    ax.set_title("Cuss Word Distribution by Gender")
    st.pyplot(fig)

else:
    st.write("Please upload a video file to process.")
