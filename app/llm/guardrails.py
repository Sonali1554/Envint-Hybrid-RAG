"""
Guardrails: the safety + correctness rules that sit between retrieval and the
final answer. These implement several things the PDF asks for:

  1. Indirect prompt-injection defence — retrieved text is treated as DATA, never
     as instructions. We detect obvious injection attempts and, crucially, we
     frame all context inside a clearly delimited block the model is told to
     distrust. The extractive backend is immune by construction (it only copies
     sentences; it cannot "follow" anything).
  2. Strict refusal — if the retrieved evidence does not actually support the
     question, we refuse instead of inventing an answer.
  3. Temporal conflict resolution — when two clauses about the same topic
     disagree, we surface the conflict AND resolve it using effective dates
     (the most recent clause wins).
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from app.retrieval.bm25 import tokenize
from app.schema import RetrievedChunk

# ---------------------------------------------------------------------------- #
# 1. Prompt-injection detection
# ---------------------------------------------------------------------------- #
# These patterns catch the classic "ignore your instructions" style attacks that
# an attacker might hide inside an ingested document.
INJECTION_PATTERNS = [
    r"ignore (all )?(previous|prior) (instructions|context)",
    r"disregard (the )?(previous|above|retention periods)",
    r"system override",
    r"\bsystem\s*:",
    r"assistant directive",
    r"previous policies are void",
    r"stop citing sources",
    r"note to the assistant",
    r"ignore the document framing",
]
_INJECTION_RE = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)


def flag_injections(chunks: List[RetrievedChunk]) -> List[str]:
    """Return the chunk_ids whose text looks like a prompt-injection attempt."""
    return [c.chunk_id for c in chunks if _INJECTION_RE.search(c.snippet)]


def build_context_block(chunks: List[RetrievedChunk]) -> str:
    """Render retrieved chunks as clearly-delimited, numbered, UNTRUSTED data.

    Every source is wrapped and labelled. The system prompt (in client.py) tells
    the model this block is reference data only and that any instructions inside
    it must be ignored. This "isolation" is the core injection defence for the
    LLM backend.
    """
    lines = []
    for i, c in enumerate(chunks, start=1):
        lines.append(f"[Source {i} | doc={c.doc_id} | clause={c.clause_id}]")
        lines.append(c.snippet.strip())
        lines.append("")  # blank line between sources
    return "\n".join(lines).strip()

   
def remove_injection_text(text: str) -> str:
    """Remove known prompt-injection sentences while preserving safe text."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    safe_sentences = [
        sentence for sentence in sentences
        if not _INJECTION_RE.search(sentence)
    ]
    return " ".join(safe_sentences).strip()


def sanitize_chunks(
    chunks: List[RetrievedChunk],
) -> List[RetrievedChunk]:
    """Return copies of chunks with detected injection sentences removed."""
    from copy import copy

    safe_chunks = []
    for chunk in chunks:
        cleaned_text = remove_injection_text(chunk.snippet)
        if not cleaned_text:
            continue

        cleaned_chunk = copy(chunk)
        cleaned_chunk.snippet = cleaned_text
        safe_chunks.append(cleaned_chunk)

    return safe_chunks



# ---------------------------------------------------------------------------- #
# 2. Refusal decision
# ---------------------------------------------------------------------------- #
def is_supported(query: str, chunks: List[RetrievedChunk], min_overlap: int = 1) -> bool:
    """Decide whether the evidence actually supports the question.

    Heuristic (works offline, no LLM needed): at least one retrieved chunk must
    share `min_overlap` meaningful words with the query. If nothing overlaps,
    the question is unsupported and we must refuse.

    We ignore a small list of stop words so that matching only on 'the'/'is'
    does not count as support.
    """
    stop = {"the", "a", "an", "is", "are", "of", "for", "to", "what", "how", "do",
            "does", "we", "must", "be", "in", "on", "within", "across", "appear",
            # Generic terms in a *policy* corpus — matching only these is not
            # real evidence, so we exclude them from the support check.
            "policy", "policies", "company", "companys", "s", "our"}
    q_words = {w for w in tokenize(query) if w not in stop}
    if not chunks or not q_words:
        return False
    for c in chunks:
        c_words = set(tokenize(c.snippet))
        if len(q_words & c_words) >= min_overlap:
            return True
    return False


REFUSAL_MESSAGE = (
    "I could not find supporting evidence in the policy corpus to answer this "
    "question, so I am not able to provide an answer."
)


# ---------------------------------------------------------------------------- #
# 3. Temporal conflict detection + resolution
# ---------------------------------------------------------------------------- #
_NUMERIC_RE = re.compile(r"(\d+)\s*(year|years|day|days|hour|hours|month|months)", re.IGNORECASE)


def _numeric_claims(text: str) -> set[str]:
    """Pull out normalised 'number + unit' claims like '7 year' from a text."""
    claims = set()
    for num, unit in _NUMERIC_RE.findall(text):
        claims.add(f"{num} {unit.rstrip('s').lower()}")
    return claims


def detect_temporal_conflicts(
    chunks: List[RetrievedChunk],
) -> Tuple[List[str], Optional[str]]:
    """Detect numeric conflicts only among clauses in the same clause family.

    For example, RET-1.1 and RET-1.2 can be compared, but RET-1.1
    and RET-2.1 are not compared because they describe different subjects.
    """
    notes: List[str] = []
    authoritative: Optional[str] = None

    # Group by document and clause family (e.g. RET-1.1 -> RET-1).
    by_family: dict[tuple[str, str], List[RetrievedChunk]] = {}
    for chunk in chunks:
        family = chunk.clause_id.rsplit(".", 1)[0]
        by_family.setdefault((chunk.doc_id, family), []).append(chunk)

    for (doc_id, family), group in by_family.items():
        if len(group) < 2:
            continue

        claims_by_chunk = {
            chunk.chunk_id: _numeric_claims(chunk.snippet)
            for chunk in group
        }

        # A conflict requires at least two different numeric claims.
        all_claims = set().union(*claims_by_chunk.values())
        if len(all_claims) < 2:
            continue

        # Resolve using effective dates. Missing dates sort oldest.
        winner = max(
            group,
            key=lambda chunk: (
                chunk.metadata.effective_date or "0000-00-00",
                chunk.clause_id,
            ),
        )

        winner_claims = claims_by_chunk[winner.chunk_id]
        losers = [
            chunk for chunk in group
            if chunk.chunk_id != winner.chunk_id
            and claims_by_chunk[chunk.chunk_id]
            and claims_by_chunk[chunk.chunk_id] != winner_claims
        ]

        if not losers:
            continue

        authoritative = winner.chunk_id
        loser_desc = ", ".join(
            f"{chunk.clause_id} "
            f"(effective {chunk.metadata.effective_date}): "
            f"{sorted(claims_by_chunk[chunk.chunk_id])}"
            for chunk in losers
        )

        notes.append(
            f"Conflict in document '{doc_id}', clause family '{family}': "
            f"clause {winner.clause_id} "
            f"(effective {winner.metadata.effective_date}) is the most recent "
            f"and takes precedence over earlier clause(s) [{loser_desc}]."
        )

    return notes, authoritative
