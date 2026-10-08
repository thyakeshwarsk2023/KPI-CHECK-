"""
src/features/embed_helper.py

Shared semantic embedding module for both training pipeline (build_features.py)
and real-time inference (app.py).
Ensures feature space consistency by using 'sentence-transformers/all-MiniLM-L6-v2'
via fastembed / sentence-transformers with graceful TF-IDF fallback.
"""

import logging
from typing import List, Tuple
import numpy as np

logger = logging.getLogger(__name__)

_GLOBAL_EMBEDDER = None


def get_embedder():
    """Initializes and returns a cached text embedder instance."""
    global _GLOBAL_EMBEDDER
    if _GLOBAL_EMBEDDER is not None:
        return _GLOBAL_EMBEDDER

    # 1. Try fastembed (ONNX runtime for all-MiniLM-L6-v2)
    try:
        from fastembed import TextEmbedding
        logger.info("Initializing fastembed ('sentence-transformers/all-MiniLM-L6-v2')...")
        _GLOBAL_EMBEDDER = ("fastembed", TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2"))
        return _GLOBAL_EMBEDDER
    except Exception as e_fast:
        logger.debug(f"fastembed not active ({e_fast}), checking sentence-transformers...")

    # 2. Try sentence-transformers
    try:
        from sentence_transformers import SentenceTransformer
        logger.info("Initializing sentence-transformers ('all-MiniLM-L6-v2')...")
        _GLOBAL_EMBEDDER = ("sentence_transformers", SentenceTransformer("all-MiniLM-L6-v2"))
        return _GLOBAL_EMBEDDER
    except Exception as e_st:
        logger.warning(f"sentence-transformers unavailable ({e_st}), using TF-IDF cosine similarity fallback.")
        _GLOBAL_EMBEDDER = ("fallback", None)
        return _GLOBAL_EMBEDDER


def compute_pair_similarity(sentence: str, line_item: str, embedder=None) -> float:
    """Computes exact cosine similarity between a single sentence and line item."""
    if embedder is None:
        embedder = get_embedder()

    backend, model = embedder

    if backend == "fastembed":
        embs = list(model.embed([sentence, line_item]))
        v1, v2 = np.array(embs[0]), np.array(embs[1])
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 > 0 and norm2 > 0:
            return float(np.dot(v1, v2) / (norm1 * norm2))
        return 0.0

    elif backend == "sentence_transformers":
        v1 = model.encode(sentence, normalize_embeddings=True)
        v2 = model.encode(line_item, normalize_embeddings=True)
        return float(np.dot(v1, v2))

    else:
        # Fallback character/token overlap normalized
        from rapidfuzz import fuzz
        return float(fuzz.token_set_ratio(sentence, line_item) / 100.0 * 0.4)


def compute_batch_similarities(sentences: List[str], line_items: List[str], embedder=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes pairwise cosine similarities and concatenated embeddings for a batch.
    Returns:
        cos_sims: 1D np.ndarray of shape (N,)
        raw_embs: 2D np.ndarray of shape (N, dim*2)
    """
    if embedder is None:
        embedder = get_embedder()

    backend, model = embedder

    if backend == "fastembed":
        logger.info("Batch encoding using fastembed ('sentence-transformers/all-MiniLM-L6-v2')...")
        sent_embs = np.array(list(model.embed(sentences)))
        line_embs = np.array(list(model.embed(line_items)))
        
        sent_norms = np.linalg.norm(sent_embs, axis=1, keepdims=True)
        line_norms = np.linalg.norm(line_embs, axis=1, keepdims=True)
        sent_normed = sent_embs / np.maximum(sent_norms, 1e-9)
        line_normed = line_embs / np.maximum(line_norms, 1e-9)
        
        cos_sims = np.sum(sent_normed * line_normed, axis=1)
        raw_embs = np.hstack([sent_normed, line_normed])
        return cos_sims, raw_embs

    elif backend == "sentence_transformers":
        logger.info("Batch encoding using sentence-transformers ('all-MiniLM-L6-v2')...")
        sent_embs = model.encode(sentences, show_progress_bar=False, normalize_embeddings=True)
        line_embs = model.encode(line_items, show_progress_bar=False, normalize_embeddings=True)
        cos_sims = np.sum(sent_embs * line_embs, axis=1)
        raw_embs = np.hstack([sent_embs, line_embs])
        return cos_sims, raw_embs

    else:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        logger.info("Batch encoding using TF-IDF cosine similarity fallback...")
        vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        vec.fit(sentences + line_items)
        s_mat = vec.transform(sentences)
        l_mat = vec.transform(line_items)
        sims = [cosine_similarity(s_mat[i], l_mat[i])[0][0] for i in range(len(sentences))]
        raw_embs = np.hstack([s_mat.toarray()[:, :100], l_mat.toarray()[:, :100]])
        return np.array(sims), raw_embs
