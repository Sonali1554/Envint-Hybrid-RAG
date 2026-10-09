"""
Loading the sample corpus from disk into our Chunk objects.

For Option B (retrieval benchmarking) the corpus is supplied as pre-chunked
JSON so that every chunk has a stable id we can use as a ground-truth label in
the evaluation. (Parsing raw PDF/DOCX/PPTX into chunks is the focus of Option C,
not Option B, so we deliberately keep ingestion simple here.)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from app.config import settings
from app.schema import Chunk


def load_corpus(path: str | None = None) -> List[Chunk]:
    """Read corpus.json and validate every record against the Chunk schema.

    Pydantic validation here is our "malformed input" guard: a record missing a
    required field (e.g. clause_id) raises a clear error instead of silently
    corrupting the index.
    """
    corpus_path = Path(path or settings.corpus_path)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    raw = json.loads(corpus_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Corpus file must contain a JSON list of chunks.")

    return [Chunk(**record) for record in raw]
