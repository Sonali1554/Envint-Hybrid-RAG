"""
Tier 1 building block: VECTOR (semantic) search using Qdrant.

Idea: we embed every chunk once and store the vectors in Qdrant. At query time
we embed the question and ask Qdrant for the nearest vectors (cosine similarity).
"Nearest" means "closest in meaning", so this finds relevant text even when the
wording is different.

Qdrant can run two ways (chosen in config.py):
  * in-memory inside this process  -> great for dev and tests, nothing to install
  * as a container service          -> used by docker-compose (QDRANT_URL set)
"""

from __future__ import annotations

from typing import List, Tuple

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from app.config import settings
from app.embeddings import get_embedder
from app.schema import Chunk


class VectorStore:
    """Thin wrapper around a Qdrant collection of policy chunks."""

    def __init__(self):
        # Connect to a real Qdrant if a URL is configured, else spin up an
        # in-process in-memory instance (no server needed).
        if settings.qdrant_url:
            self.client = QdrantClient(url=settings.qdrant_url)
        else:
            self.client = QdrantClient(location=":memory:")

        self.collection = settings.qdrant_collection
        self.embedder = get_embedder()
        # Qdrant point ids must be ints/UUIDs, but our chunk ids are strings
        # like "SEC-1.1". We keep a simple int<->chunk_id mapping.
        self._id_to_chunk: dict[int, str] = {}

    def index(self, chunks: List[Chunk]) -> None:
        """Embed all chunks and (re)create the collection with their vectors.

        This is idempotent: we recreate the collection from scratch each time,
        so running it twice leaves exactly one copy of each chunk.
        """
        # Fresh collection sized to our embedding dimension, cosine distance.
        self.client.recreate_collection(
            collection_name=self.collection,
            vectors_config=qm.VectorParams(
                size=settings.embed_dim, distance=qm.Distance.COSINE
            ),
        )

        # Empty corpus -> nothing to embed or upsert. Leave the (empty)
        # collection in place so searches simply return no results.
        if not chunks:
            return

        texts = [c.text for c in chunks]
        vectors = self.embedder.encode(texts)

        points = []
        for i, chunk in enumerate(chunks):
            self._id_to_chunk[i] = chunk.chunk_id
            points.append(
                qm.PointStruct(
                    id=i,
                    vector=vectors[i].tolist(),
                    # The payload IS our explicit schema + the raw text.
                    payload={"chunk_id": chunk.chunk_id, **chunk.metadata.model_dump()},
                )
            )
        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, query: str, k: int) -> List[Tuple[str, float]]:
        """Return the k most semantically similar chunks as (chunk_id, score)."""
        query_vec = self.embedder.encode([query])[0]
        hits = self.client.search(
            collection_name=self.collection,
            query_vector=query_vec.tolist(),
            limit=k,
        )
        # score is cosine similarity in [-1, 1]; higher is better.
        return [(h.payload["chunk_id"], float(h.score)) for h in hits]
