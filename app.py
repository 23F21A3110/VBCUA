import os
import tempfile

import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt

from utils.fluency import fluency_score
from utils.concept import concept_score


# -------------------------------
# App Title
# -------------------------------
st.title("🎤 Voice-Based Concept Understanding Analyser")


# -------------------------------
# Audio Upload
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload your audio",
    type=["wav", "mp3"]
)


# -------------------------------
# Process Audio
# -------------------------------
if uploaded_file:

    # Save uploaded audio temporarily
    ext = os.path.splitext(uploaded_file.name)[1]

    temp = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=ext
    )

    temp.write(uploaded_file.read())
    temp.close()

    audio_path = temp.name

    st.success("✅ Audio uploaded successfully!")

    # -------------------------------
    # Audio Player
    # -------------------------------
    st.subheader("🎵 Uploaded Audio")
    st.audio(uploaded_file)

    # -------------------------------
    # Waveform
    # -------------------------------
    st.subheader("📈 Audio Waveform")

    y, sr = librosa.load(audio_path)

    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax)

    ax.set_title("Audio Waveform")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")

    st.pyplot(fig)

    # -------------------------------
    # Speech Analysis
    # -------------------------------
    metrics = fluency_score(audio_path)

    st.subheader("🗣️ Speech Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Duration",
            f"{metrics['duration']} sec"
        )

    with col2:
        st.metric(
            "Energy",
            metrics["energy"]
        )

    with col3:
        st.metric(
            "Pause Ratio",
            metrics["pause_ratio"]
        )

    # -------------------------------
    # Concept Understanding
    # -------------------------------
    text, score = concept_score(audio_path)

    st.subheader("🧠 Concept Understanding")

    st.write("### Transcribed Text")

    st.success(text)

    st.metric(
        "Concept Score",
        f"{score:.2f}%"
    )

    # Progress Bar
    st.progress(int(score))