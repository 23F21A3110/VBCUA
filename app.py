import whisper
import google.generativeai as genai
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

model = genai.GenerativeModel("gemini-2.5-flash")

def generate_feedback(transcribed_text):
    prompt = f"""
    Student Explanation:

    {transcribed_text}
from utils.fluency import fluency_score

from utils.concept import concept_score, generate_feedback, generate_summary
from utils.pdf_report import create_pdf
# -------------------------------
# App Title
# -------------------------------
st.title("🎤 Voice-Based Concept Understanding Analyser")

    Give feedback in this format:

    Strengths:
    Weaknesses:
    Suggestions:
    Overall Rating (out of 10):
    """

    response = model.generate_content(prompt)
    return response.text
def generate_summary(transcribed_text):
    prompt = f"""
    Summarize the following student's explanation in 3-4 sentences.

    Explanation:
    {transcribed_text}
    """

    response = model.generate_content(prompt)
    return response.text
whisper_model = whisper.load_model("base")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

REFERENCE = """
Machine learning is a branch of artificial intelligence
that allows computers to learn from data.
"""

def concept_score(audio_path):

    result = whisper_model.transcribe(audio_path)

    text = result["text"]

    v1 = embed_model.encode([text])
    v2 = embed_model.encode([REFERENCE])

    similarity = cosine_similarity(v1, v2)[0][0]

    return text, round(float(similarity * 100), 2)
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
