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
