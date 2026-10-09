"""
Turning text into vectors (embeddings).

Semantic search works by converting text into a list of numbers (a "vector")
such that passages with similar meaning end up close together. This file gives
us ONE function, `get_embedder()`, that returns an object with an `.encode()`
method — the rest of the app never cares which backend is behind it.

Two backends:
  * RealEmbedder  -> the open-source BAAI/bge-small-en-v1.5 model (free, accurate).
  * FakeEmbedder  -> a deterministic, offline hash-based vector. No model download,
                     no internet. Used in tests/CI so they run anywhere for free.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import List

import numpy as np

from app.config import settings


class FakeEmbedder:
    """A stand-in embedder that needs no model and no internet.

    It hashes each word into a fixed slot of the vector. It is NOT good at
    semantics, but it is 100% deterministic and dependency-free, which is
    exactly what automated tests want. We normalise the vector so that cosine
    similarity behaves the same way it does for the real model.
    """

    def __init__(self, dim: int):
        self.dim = dim

    def encode(self, texts: List[str], **_: object) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in text.lower().split():
                # Map the word to a slot using a stable hash, then add 1 there.
                h = int(hashlib.md5(word.encode()).hexdigest(), 16)
                vectors[row, h % self.dim] += 1.0
            # Normalise to unit length so cosine similarity == dot product.
            norm = np.linalg.norm(vectors[row])
            if norm > 0:
                vectors[row] /= norm
        return vectors


class RealEmbedder:
    """Wraps the real sentence-transformers model.

    We import sentence-transformers lazily (inside __init__) so that simply
    importing this file never triggers a big model download.
    """

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str], **kwargs: object) -> np.ndarray:
        # normalize_embeddings=True -> unit vectors, so cosine == dot product.
        return np.asarray(
            self.model.encode(texts, normalize_embeddings=True, **kwargs),
            dtype=np.float32,
        )


@lru_cache(maxsize=1)
def get_embedder():
    """Return the configured embedder, building it only once (cached)."""
    if settings.embed_backend == "fake":
        return FakeEmbedder(settings.embed_dim)
    return RealEmbedder(settings.embed_model)
