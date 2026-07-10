import os
import tempfile

import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt

from utils.fluency import fluency_score

from utils.concept import concept_score, generate_feedback, generate_summary
from utils.pdf_report import create_pdf
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
    # -------------------------------
# AI Summary
# -------------------------------
    summary = generate_summary(text)

    st.subheader("📄 AI Summary")
    st.write(summary)

    st.metric(
        "Concept Score",
        f"{score:.2f}%"
    )

    # Progress Bar
    st.progress(int(score))
    # -------------------------------
    # Understanding Level
    # -------------------------------
    if score >= 70:
     st.success("✅ Strong Understanding")
    elif score >= 40:
     st.warning("🟡 Moderate Understanding")
    else:
      st.error("🔴 Poor Understanding")
    # -------------------------------
# AI Feedback
# -------------------------------
    feedback = generate_feedback(text)

    st.subheader("📝 AI Feedback")
    st.write(feedback)
    # -------------------------------
# Overall Performance Score
# -------------------------------

  # Calculate Fluency Score
    fluency = (100 - metrics["pause_ratio"] * 100)

# Limit fluency score to 100
    fluency = max(0, min(100, fluency))

# Calculate Overall Score
    overall_score = (0.7 * score) + (0.3 * fluency)

    st.subheader("🏆 Overall Performance")

    st.metric(
    "Overall Score",
    f"{overall_score:.2f}%"
     )

    st.progress(int(overall_score))
    # -------------------------------
# Generate PDF Report
# -------------------------------
    create_pdf(text, summary, score, feedback)

    with open("report.pdf", "rb") as pdf_file:
     st.download_button(
        label="📥 Download PDF Report",
        data=pdf_file,
        file_name="VBCUA_Report.pdf",
        mime="application/pdf"
    )
