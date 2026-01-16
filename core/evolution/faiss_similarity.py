"""FAISS helpers for cosine similarity computations."""

from __future__ import annotations

import numpy as np
import faiss


def build_ip_index(vectors: np.ndarray) -> faiss.Index:
    """Build an IndexFlatIP assuming vectors are L2-normalized."""
    if vectors.dtype != np.float32:
        vectors = vectors.astype(np.float32)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def all_pairs_cosine(vectors: np.ndarray) -> np.ndarray:
    """Compute clamped cosine similarity matrix for normalized vectors."""
    if vectors.dtype != np.float32:
        vectors = vectors.astype(np.float32)
    index = build_ip_index(vectors)
    sims = faiss.vector_to_array(index.compute_inner_products(vectors))
    sims = sims.reshape(len(vectors), len(vectors))
    sims = np.clip((sims + 1.0) / 2.0, 0.0, 1.0)
    return sims
