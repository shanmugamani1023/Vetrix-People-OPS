"""
Retriever: builds FAISS index from chunks, filters by eligibility, applies token budget.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

import numpy as np
import structlog
import tiktoken

from app.models import Chunk, Employee, ManifestDoc

if TYPE_CHECKING:
    pass

log = structlog.get_logger()

TOKENIZER = tiktoken.get_encoding("cl100k_base")
MAX_CORPUS_TOKENS = 3000

# Authority rank for precedence: higher = more authoritative
_AUTHORITY = {"policy": 2, "addendum": 2, "memo": 2, "handbook": 2, "calendar": 2, "faq": 1, "wiki": 1}

# Access level hierarchy
_ACCESS_RANK = {"all": 0, "managers": 1, "hr": 2}
_ROLE_ACCESS = {"employee": {"all"}, "manager": {"all", "managers"}, "hr": {"all", "managers", "hr"}}


def count_tokens(text: str) -> int:
    return len(TOKENIZER.encode(text))


# ── Eligibility & precedence ──────────────────────────────────────────────────

def filter_eligible_docs(
    manifest_docs: list[ManifestDoc],
    employee: Employee,
    reference_date: date,
) -> list[ManifestDoc]:
    """
    Return docs the employee can read that are effective on reference_date.
    Rules (spec §5.1):
    - access_level must be in the employee's role access set
    - regions must include employee's region or GLOBAL
    - effective_date ≤ reference_date
    """
    allowed_levels = _ROLE_ACCESS.get(employee.role, {"all"})
    result = []
    for doc in manifest_docs:
        if doc.access_level not in allowed_levels:
            continue
        if "GLOBAL" not in doc.regions and employee.region not in doc.regions:
            continue
        if date.fromisoformat(doc.effective_date) > reference_date:
            continue
        result.append(doc)
    return result


def detect_upcoming_changes(
    manifest_docs: list[ManifestDoc],
    employee: Employee,
    reference_date: date,
) -> list[ManifestDoc]:
    """
    Docs that are not yet effective but are readable by the employee
    and would win on their effective date. Used to mention upcoming changes.
    """
    allowed_levels = _ROLE_ACCESS.get(employee.role, {"all"})
    upcoming = []
    for doc in manifest_docs:
        if doc.access_level not in allowed_levels:
            continue
        if "GLOBAL" not in doc.regions and employee.region not in doc.regions:
            continue
        eff = date.fromisoformat(doc.effective_date)
        if eff <= reference_date:
            continue
        upcoming.append(doc)
    return upcoming


def precedence_key(doc: ManifestDoc) -> tuple:
    """Higher tuple = higher precedence (for sorting)."""
    authority = _AUTHORITY.get(doc.doc_type, 0)
    scope = 0 if "GLOBAL" in doc.regions else 1  # specific region > GLOBAL
    return (authority, scope, doc.effective_date)


# ── Retriever class ───────────────────────────────────────────────────────────

class Retriever:
    def __init__(self, embed_model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = embed_model_name
        self._model = None
        self._index = None          # faiss.IndexFlatIP
        self._chunk_ids: list[int] = []      # FAISS row → chunk_id in DB
        self._chunk_map: dict[int, Chunk] = {}  # chunk_id → Chunk
        self._ready = False

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def build_index(self, chunks: list[Chunk], chunk_ids: list[int]) -> None:
        """
        Build FAISS IndexFlatIP from chunk embeddings.
        chunk_ids must align positionally with chunks.
        """
        import faiss

        if not chunks:
            log.warning("no_chunks_to_index")
            self._ready = True
            return

        model = self._get_model()
        texts = [c.text for c in chunks]
        log.info("embedding_chunks", count=len(texts))
        embeddings = model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
        embeddings = np.array(embeddings, dtype="float32")

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        self._index = index
        self._chunk_ids = chunk_ids
        self._chunk_map = {cid: c for cid, c in zip(chunk_ids, chunks)}
        self._ready = True
        log.info("faiss_index_built", vectors=len(chunk_ids), dim=dim)

    @property
    def ready(self) -> bool:
        return self._ready

    def retrieve(
        self,
        query: str,
        employee: Employee,
        reference_date: date,
        manifest_docs: list[ManifestDoc],
        max_tokens: int = MAX_CORPUS_TOKENS,
    ) -> list[Chunk]:
        """
        1. Filter eligible doc_ids
        2. Vector search
        3. Filter results to eligible docs
        4. Greedy token-budget selection
        """
        if not self._ready or self._index is None:
            return []

        eligible_docs = filter_eligible_docs(manifest_docs, employee, reference_date)
        eligible_doc_ids = {d.doc_id for d in eligible_docs}

        if not eligible_doc_ids:
            return []

        model = self._get_model()
        q_emb = model.encode([query], normalize_embeddings=True)
        q_emb = np.array(q_emb, dtype="float32")

        # Search top-k (enough to have eligible ones after filtering)
        k = min(len(self._chunk_ids), max(50, len(self._chunk_ids)))
        scores, indices = self._index.search(q_emb, k)

        selected: list[Chunk] = []
        used_tokens = 0

        for idx in indices[0]:
            if idx < 0 or idx >= len(self._chunk_ids):
                continue
            cid = self._chunk_ids[idx]
            chunk = self._chunk_map.get(cid)
            if chunk is None:
                continue
            if chunk.doc_id not in eligible_doc_ids:
                continue
            ct = chunk.token_count or count_tokens(chunk.text)
            if used_tokens + ct > max_tokens:
                continue
            selected.append(chunk)
            used_tokens += ct

        log.info("retrieval_done", eligible_docs=len(eligible_doc_ids),
                 chunks_returned=len(selected), corpus_tokens=used_tokens)
        return selected

    def total_corpus_tokens(self, chunks: list[Chunk]) -> int:
        return sum(c.token_count or count_tokens(c.text) for c in chunks)
