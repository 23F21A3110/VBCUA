import whisper
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

whisper_model = whisper.load_model("base")

embed_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

REFERENCE = """
Machine learning is a branch of artificial intelligence
that allows computers to learn from data.
"""

def concept_score(audio_path):

    result = whisper_model.transcribe(
        audio_path
    )

    text = result["text"]

    v1 = embed_model.encode([text])
    v2 = embed_model.encode([REFERENCE])

    similarity = cosine_similarity(
        v1,
        v2
    )[0][0]

    return (
        text,
        round(float(similarity * 100), 2)
    )