import tempfile
from pathlib import Path

import streamlit as st
import google.generativeai as genai

from utils.concept import concept_score, generate_feedback, generate_summary
from utils.fluency import fluency_score
from utils.pdf_report import create_pdf


def save_audio_file(file_bytes: bytes, suffix: str) -> str:
    if not suffix.startswith("."):
        suffix = f".{suffix}"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(file_bytes)
        return temp_file.name


def display_metrics(metrics: dict) -> None:
    col1, col2, col3 = st.columns(3)
    col1.metric("Duration", f"{metrics['duration']} sec")
    col2.metric("Energy", metrics["energy"])
    col3.metric("Pause Ratio", metrics["pause_ratio"])


def understanding_label(score: float) -> tuple[str, str]:
    if score >= 70:
        return "✅ Strong Understanding", "success"
    if score >= 40:
        return "🟡 Moderate Understanding", "warning"
    return "🔴 Poor Understanding", "error"


def main() -> None:
    st.set_page_config(
        page_title="Voice-Based Concept Understanding Analyzer",
        page_icon="🎤",
        layout="wide",
    )

    st.title("🎤 Voice-Based Concept Understanding Analyzer")
    st.write(
        "Upload a student audio explanation to generate a transcript, score, summary, feedback, and PDF report."
    )

    uploaded_file = st.file_uploader(
        "Upload audio file",
        type=["wav", "mp3", "m4a", "flac", "ogg"],
    )

    if not uploaded_file:
        st.info("Please upload an audio file to begin.")
        return

    audio_bytes = uploaded_file.read()
    st.audio(audio_bytes, format=uploaded_file.type)

    if st.button("Analyze"):
        suffix = Path(uploaded_file.name).suffix or ".wav"
        audio_path = save_audio_file(audio_bytes, suffix)

        try:
            genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
            model = genai.GenerativeModel("gemini-2.5-flash")

            with st.spinner("Analyzing audio and generating feedback..."):
                metrics = fluency_score(audio_path)
                text, score = concept_score(audio_path)
                summary = generate_summary(model, text)
                feedback = generate_feedback(model, text)
                fluency_score_value = max(0, min(100, 100 - metrics["pause_ratio"] * 100))
                overall_score = (0.7 * score) + (0.3 * fluency_score_value)

            st.subheader("🗣️ Speech Analysis")
            display_metrics(metrics)

            st.subheader("🧠 Concept Understanding")
            st.write("### Transcribed Text")
            st.success(text)
            st.metric("Concept Score", f"{score:.2f}%")
            st.progress(min(int(score), 100))

            label, status = understanding_label(score)
            if status == "success":
                st.success(label)
            elif status == "warning":
                st.warning(label)
            else:
                st.error(label)

            st.subheader("📄 AI Summary")
            st.write(summary)

            st.subheader("📝 AI Feedback")
            st.write(feedback)

            st.subheader("🏆 Overall Performance")
            st.metric("Overall Score", f"{overall_score:.2f}%")
            st.progress(min(int(overall_score), 100))

            create_pdf(text, summary, score, feedback)
            with open("report.pdf", "rb") as pdf_file:
                st.download_button(
                    label="📥 Download PDF Report",
                    data=pdf_file,
                    file_name="VBCUA_Report.pdf",
                    mime="application/pdf",
                )
        except Exception as error:
            st.error(f"Analysis failed: {error}")
        finally:
            try:
                Path(audio_path).unlink()
            except OSError:
                pass


if __name__ == "__main__":
    main()

