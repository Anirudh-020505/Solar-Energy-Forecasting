"""
FAISS vector store for the RAG knowledge base.

Uses IndexFlatIP (inner product on L2-normalised embeddings = cosine similarity).
The index is built once at startup and held in memory — no persistence needed
since rebuild takes < 2 seconds on the full 4-document knowledge base.

Usage
-----
    from rag.knowledge_base import load_and_chunk_documents
    from rag.faiss_store import FAISSStore

    store = FAISSStore()
    store.build(load_and_chunk_documents())
    results = store.query("battery storage ramp rate", k=6)
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    faiss = None  # type: ignore[assignment]
    _FAISS_AVAILABLE = False

from rag.embedder import embed_texts, get_embedder


class FAISSStore:
    """
    In-memory FAISS index over document chunks.

    Attributes
    ----------
    chunks   : list of chunk dicts (text, source, chunk_id) — set by build()
    _index   : faiss.IndexFlatIP — cosine similarity index
    _built   : bool
    """

    def __init__(self) -> None:
        self.chunks: list[dict[str, Any]] = []
        self._index: Any = None  # faiss.IndexFlatIP
        self._built: bool = False
        self._embedder = None  # lazy-loaded on first build/query

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, chunks: list[dict[str, Any]]) -> None:
        """
        Embed all chunks and populate the FAISS index.

        Parameters
        ----------
        chunks : list of dicts returned by load_and_chunk_documents()
                 Each dict must have at least a "text" key.

        Raises
        ------
        RuntimeError if faiss-cpu is not installed.
        ValueError   if chunks is empty.
        """
        if not chunks:
            raise ValueError("Cannot build FAISSStore from an empty chunk list.")

        if not _FAISS_AVAILABLE:
            raise RuntimeError(
                "faiss-cpu is not installed. Run: pip install faiss-cpu"
            )

        self._embedder = get_embedder()
        self.chunks = chunks

        texts = [c["text"] for c in chunks]
        embeddings = embed_texts(texts, self._embedder)  # (N, 384) float32, L2-normed

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # cosine sim via inner product on normed vecs
        index.add(embeddings)

        self._index = index
        self._built = True

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        k: int = 6,
    ) -> list[dict[str, Any]]:
        """
        Retrieve the top-k most relevant chunks for *query_text*.

        Parameters
        ----------
        query_text : natural-language query string
        k          : number of results to return (capped at index size)

        Returns
        -------
        List of dicts, each containing:
            text      str   — chunk text
            source    str   — originating document filename
            chunk_id  str   — unique chunk identifier
            score     float — cosine similarity score (0–1; higher = more relevant)

        Falls back to keyword overlap ranking when FAISS is unavailable.
        """
        if not query_text.strip():
            return []

        if not self._built:
            raise RuntimeError(
                "FAISSStore has not been built yet. Call .build(chunks) first."
            )

        actual_k = min(k, len(self.chunks))

        if not _FAISS_AVAILABLE:
            return self._keyword_fallback(query_text, actual_k)

        # Embed and normalise the query
        query_vec = embed_texts([query_text], self._embedder)  # (1, 384)

        scores, indices = self._index.search(query_vec, actual_k)

        results: list[dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for padding when k > index size
                continue
            chunk = dict(self.chunks[idx])
            chunk["score"] = round(float(score), 4)
            results.append(chunk)

        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _keyword_fallback(
        self,
        query_text: str,
        k: int,
    ) -> list[dict[str, Any]]:
        """
        Simple keyword-overlap ranking used when faiss-cpu is not installed.

        Scores each chunk by the fraction of unique query words that appear in
        the chunk text (case-insensitive). Not semantic — purely lexical.
        """
        query_words = set(query_text.lower().split())
        scored: list[tuple[float, dict[str, Any]]] = []

        for chunk in self.chunks:
            chunk_words = set(chunk["text"].lower().split())
            overlap = len(query_words & chunk_words)
            score = overlap / max(len(query_words), 1)
            scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, chunk in scored[:k]:
            result = dict(chunk)
            result["score"] = round(score, 4)
            results.append(result)

        return results

    @property
    def size(self) -> int:
        """Number of chunks currently indexed."""
        return len(self.chunks)

    def __repr__(self) -> str:
        status = f"built, {self.size} chunks" if self._built else "not built"
        return f"FAISSStore({status})"
