"""
Sentence-Transformers embedding wrapper.

get_embedder() is decorated with @st.cache_resource so the model is loaded
once per Streamlit process and reused across all pages and reruns.

Outside Streamlit (tests, CLI), call get_embedder() directly — without the
cache decorator it simply returns a fresh SentenceTransformer instance.
"""

from __future__ import annotations

import numpy as np

try:
    import streamlit as st
    _STREAMLIT_AVAILABLE = True
except ImportError:
    _STREAMLIT_AVAILABLE = False

from sentence_transformers import SentenceTransformer

# Model chosen for: 22 MB footprint, no GPU required, 384-dim embeddings,
# strong retrieval performance on domain-specific short passages.
_MODEL_NAME = "all-MiniLM-L6-v2"


def _load_embedder() -> SentenceTransformer:
    return SentenceTransformer(_MODEL_NAME)


if _STREAMLIT_AVAILABLE:
    @st.cache_resource(show_spinner="Loading embedding model...")
    def get_embedder() -> SentenceTransformer:
        """
        Return a cached SentenceTransformer instance (Streamlit context).

        The model is downloaded on first call (~22 MB) and cached for the
        lifetime of the Streamlit server process.
        """
        return _load_embedder()
else:
    def get_embedder() -> SentenceTransformer:  # type: ignore[misc]
        """Return a SentenceTransformer instance (non-Streamlit context)."""
        return _load_embedder()


def embed_texts(texts: list[str], embedder: SentenceTransformer | None = None) -> np.ndarray:
    """
    Embed a list of strings and return L2-normalised vectors.

    Parameters
    ----------
    texts    : list of strings to embed
    embedder : optional pre-loaded model; loads one if not provided

    Returns
    -------
    np.ndarray of shape (len(texts), 384), dtype float32, L2-normalised
    so that dot-product == cosine similarity (required by IndexFlatIP).
    """
    model = embedder or get_embedder()
    vecs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    vecs = vecs.astype(np.float32)
    # L2-normalise each row so IndexFlatIP gives cosine similarity scores
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)  # avoid div-by-zero on zero vectors
    return vecs / norms
