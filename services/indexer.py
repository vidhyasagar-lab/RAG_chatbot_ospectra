import os
import pickle
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

from utils.env_utils import get_env


# ------------------------------------------------------------------
# Environment configuration
# ------------------------------------------------------------------
STORAGE_DIR = get_env("STORAGE_DIR")

CHUNKS_PATH = f"{STORAGE_DIR}/chunks.pkl"
BM25_PATH = f"{STORAGE_DIR}/bm25.pkl"
TFIDF_PATH = f"{STORAGE_DIR}/tfidf.pkl"


# ------------------------------------------------------------------
# Index Builder
# ------------------------------------------------------------------
def build_indexes(chunks: list[str]) -> None:
    os.makedirs(STORAGE_DIR, exist_ok=True)

    # Store raw chunks
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    # Build BM25 index
    bm25 = BM25Okapi([chunk.split() for chunk in chunks])
    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25, f)

    # Build TF-IDF index
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(chunks)

    with open(TFIDF_PATH, "wb") as f:
        pickle.dump((vectorizer, tfidf_matrix), f)
