import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from utils.env_utils import get_env


# ------------------------------------------------------------------
# Environment configuration
# ------------------------------------------------------------------
STORAGE_DIR = get_env("STORAGE_DIR")
TOP_K = int(get_env("TOP_K_RETRIEVAL") or 10)


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------
def safe_normalize(scores: np.ndarray) -> np.ndarray:
    max_val = np.max(scores)
    if max_val == 0:
        return np.zeros_like(scores)
    return scores / max_val


# ------------------------------------------------------------------
# Hybrid Retrieval
# ------------------------------------------------------------------
def hybrid_retrieve(query: str):
    with open(f"{STORAGE_DIR}/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    with open(f"{STORAGE_DIR}/bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)

    with open(f"{STORAGE_DIR}/tfidf.pkl", "rb") as f:
        vectorizer, tfidf = pickle.load(f)

    # -------------------------
    # BM25 scores
    # -------------------------
    bm25_scores = bm25.get_scores(query.split())

    # -------------------------
    # TF-IDF cosine scores
    # -------------------------
    q_vec = vectorizer.transform([query])
    tfidf_scores = cosine_similarity(q_vec, tfidf)[0]

    # -------------------------
    # Normalization
    # -------------------------
    bm25_norm = safe_normalize(bm25_scores)
    tfidf_norm = safe_normalize(tfidf_scores)

    # -------------------------
    # Weighted merge
    # -------------------------
    scores = 0.5 * bm25_norm + 0.5 * tfidf_norm

    # -------------------------
    # Top-K selection
    # -------------------------
    top_idx = np.argsort(scores)[-TOP_K:][::-1]
    return [chunks[i] for i in top_idx]
