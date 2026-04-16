"""
Document loader and chunker for the RAG knowledge base.

Reads the four .txt files from rag/data/, splits them into overlapping
fixed-size chunks, and returns a list of dicts consumed by FAISSStore.build().

No external dependencies — pure Python + pathlib.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# Directory containing the four hand-written knowledge base documents.
_DATA_DIR = Path(__file__).parent / "data"

_KNOWN_SOURCES = [
    "grid_management.txt",
    "renewable_integration.txt",
    "battery_standards.txt",
    "demand_response.txt",
]


def load_and_chunk_documents(
    chunk_size: int = 300,
    overlap: int = 50,
    data_dir: Path | str | None = None,
) -> list[dict[str, Any]]:
    """
    Load every .txt file from *data_dir* and split into overlapping word chunks.

    Parameters
    ----------
    chunk_size : int
        Target number of words per chunk.
    overlap : int
        Number of words shared between consecutive chunks (sliding window).
    data_dir : Path or str, optional
        Override the default rag/data/ directory (useful for testing).

    Returns
    -------
    List of dicts, each with:
        text      str  — chunk text (may be slightly over chunk_size at sentence boundaries)
        source    str  — filename of the originating document
        chunk_id  str  — "{source}::chunk_{n}" unique identifier
    """
    base = Path(data_dir) if data_dir else _DATA_DIR

    if not base.exists():
        raise FileNotFoundError(f"RAG data directory not found: {base}")

    # Load preferred sources first, then any extra .txt files present
    txt_files = [base / s for s in _KNOWN_SOURCES if (base / s).exists()]
    extra = sorted(p for p in base.glob("*.txt") if p.name not in _KNOWN_SOURCES)
    txt_files.extend(extra)

    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {base}")

    all_chunks: list[dict[str, Any]] = []

    for doc_path in txt_files:
        source = doc_path.name
        text = doc_path.read_text(encoding="utf-8").strip()

        if not text:
            continue

        words = text.split()
        chunks = _sliding_window(words, chunk_size, overlap)

        for idx, chunk_words in enumerate(chunks):
            all_chunks.append({
                "text": " ".join(chunk_words),
                "source": source,
                "chunk_id": f"{source}::chunk_{idx}",
            })

    return all_chunks


def _sliding_window(
    words: list[str],
    chunk_size: int,
    overlap: int,
) -> list[list[str]]:
    """
    Split a flat word list into overlapping windows.

    If the document is shorter than one chunk, returns the entire document
    as a single chunk. The final chunk extends to the end of the document
    even if shorter than chunk_size.
    """
    if len(words) <= chunk_size:
        return [words]

    step = max(chunk_size - overlap, 1)
    chunks: list[list[str]] = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunks.append(words[start:end])
        if end >= len(words):
            break
        start += step

    return chunks
