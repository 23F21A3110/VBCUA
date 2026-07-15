import whisper
import google.generativeai as genai
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
