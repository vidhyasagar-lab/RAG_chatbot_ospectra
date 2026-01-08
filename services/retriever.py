import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from utils.env_utils import get_env
from utils.logger import logger

STORAGE_DIR = get_env("STORAGE_DIR")
TOP_K = int(get_env("TOP_K_RETRIEVAL") or 10)


def safe_normalize(scores: np.ndarray) -> np.ndarray:
    max_val = np.max(scores)
    if max_val == 0:
        return np.zeros_like(scores)
    return scores / max_val


def hybrid_retrieve(query: str):
    logger.info("Starting hybrid retrieval")

    try:
        with open(f"{STORAGE_DIR}/chunks.pkl", "rb") as f:
            chunks = pickle.load(f)

        with open(f"{STORAGE_DIR}/bm25.pkl", "rb") as f:
            bm25 = pickle.load(f)

        with open(f"{STORAGE_DIR}/tfidf.pkl", "rb") as f:
            vectorizer, tfidf = pickle.load(f)

        # BM25 scores
        bm25_scores = bm25.get_scores(query.split())

        # TF-IDF cosine scores
        q_vec = vectorizer.transform([query])
        tfidf_scores = cosine_similarity(q_vec, tfidf)[0]

        # Normalize
        bm25_norm = safe_normalize(bm25_scores)
        tfidf_norm = safe_normalize(tfidf_scores)

        # Weighted merge
        scores = 0.5 * bm25_norm + 0.5 * tfidf_norm

        # Top-K selection
        top_idx = np.argsort(scores)[-TOP_K:][::-1]

        # 🔥 RETURN (index, text) tuples
        results = [(i, chunks[i]) for i in top_idx]

        logger.info("Hybrid retrieval completed. Returned %d results", len(results))
        return results

    except Exception as e:
        logger.error("Error during hybrid retrieval: %s", str(e))
        raise


def rerank(query: str, candidates: list[tuple[int, str]]):
    logger.info("Starting reranking")

    texts = [text for _, text in candidates]

    with open(f"{STORAGE_DIR}/tfidf.pkl", "rb") as f:
        vectorizer, _ = pickle.load(f)

    q_vec = vectorizer.transform([query])
    candidate_vecs = vectorizer.transform(texts)

    scores = cosine_similarity(q_vec, candidate_vecs)[0]

    reranked = sorted(
        zip(candidates, scores),
        key=lambda x: x[1],
        reverse=True
    )

    logger.info("Reranking completed")

    # Return only text for LLM
    return [text for ((_, text), _) in reranked]
