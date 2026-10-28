"""The RAG pipeline: read -> chunk -> embed -> store -> retrieve -> generate."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

import llm_client as llm
from text_chunker import chunk_pages
from embedder import embed_texts
from pdf_loader import load_document
from vector_store import VectorStore

EMBEDDING_DIM = 384  # size of vectors from all-MiniLM-L6-v2
BATCH = 64  # chunks embedded per step
MIN_SCORE = 0.10  # below this, a chunk is probably not related to the question
RELATIVE_CUTOFF = 0.5  # keep passages scoring at least half of the best one
KEYWORD_WEIGHT = 0.25  # how much exact word matches add to the meaning score

STOPWORDS = {
    "what", "whats", "the", "and", "for", "with", "how", "why", "does", "did", "can",
    "you", "tell", "me", "explain", "about", "give", "show", "example", "examples",
    "mean", "means", "meaning", "define", "definition", "from", "use", "used", "using",
    "this", "that", "are", "was", "were", "will", "its", "which", "when", "where",
}


def keyword_overlap(question: str, text: str) -> float:
    """Share of the question's important words that appear in the text (0 to 1).

    Meaning-based search can miss very short questions like "what's VIEW?",
    so exact word matches give a small extra boost. A word also matches its
    longer forms: "view" matches "views".
    """
    terms = [t for t in re.findall(r"[a-z0-9_]+", question.lower())
             if len(t) >= 3 and t not in STOPWORDS]
    if not terms:
        return 0.0
    lowered = text.lower()
    found = sum(1 for t in terms if re.search(r"\b" + re.escape(t), lowered))
    return found / len(terms)


def rerank(question: str, candidates):
    """Sort (chunk, meaning_score) pairs by meaning score + keyword bonus."""
    scored = [
        (chunk, score + KEYWORD_WEIGHT * keyword_overlap(question, chunk.text))
        for chunk, score in candidates
    ]
    return sorted(scored, key=lambda pair: pair[1], reverse=True)


@dataclass
class Source:
    file: str
    page: int
    score: float
    preview: str


@dataclass
class Answer:
    text: str
    sources: list[Source] = field(default_factory=list)
    used_llm: bool = False
    warning: str | None = None


class StudyAssistant:
    def __init__(self, chunk_size: int = 180, overlap: int = 40):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.store = VectorStore(EMBEDDING_DIM)
        self.files: dict[str, int] = {}  # filename -> number of chunks

    def add_document(self, data: bytes, filename: str, on_progress=None) -> int:
        """Index one file. Returns how many chunks were created.

        `on_progress(done, total)` is called after every batch, so the UI can
        show a progress bar for big files.
        """
        # Always clear old chunks of this file first, even if a past run was
        # interrupted half way. This stops duplicate passages in the index.
        self.remove_document(filename)

        pages = load_document(data, filename)
        chunks = chunk_pages(pages, self.chunk_size, self.overlap)
        if not chunks:
            raise ValueError(
                f"No readable text in {filename}. It may be a scanned PDF (images only)."
            )

        # Embed in small groups so memory stays low and the UI can show progress.
        # Nothing goes into the index until every group is done, so a stopped
        # run leaves no half-indexed file behind.
        total = len(chunks)
        parts = []
        for start in range(0, total, BATCH):
            group = chunks[start:start + BATCH]
            parts.append(embed_texts([c.text for c in group]))
            if on_progress:
                on_progress(min(start + BATCH, total), total)

        self.store.add(chunks, np.vstack(parts))
        self.files[filename] = total
        return total

    def remove_document(self, filename: str) -> None:
        self.store.remove_source(filename)
        self.files.pop(filename, None)

    def ask(self, question: str, k: int = 4) -> Answer:
        question = question.strip()
        if not question:
            return Answer("Please type a question.")
        if len(self.store) == 0:
            return Answer("Upload at least one document first.")

        query_vector = embed_texts([question])[0]
        # Get extra candidates, then re-rank them with a keyword bonus.
        candidates = self.store.search(query_vector, k * 3)
        hits = [h for h in rerank(question, candidates)[:k] if h[1] >= MIN_SCORE]
        if hits:
            # Drop passages that are much weaker than the best one (just noise).
            best = hits[0][1]
            hits = [h for h in hits if h[1] >= RELATIVE_CUTOFF * best]
        if not hits:
            return Answer("I could not find this in your notes.")

        sources = [
            Source(c.source, c.page, score, c.text[:220] + ("..." if len(c.text) > 220 else ""))
            for c, score in hits
        ]

        if not llm.api_key_available():
            return Answer(llm.fallback_answer(hits), sources, used_llm=False)

        try:
            text = llm.ask_llm(question, hits)
            return Answer(text, sources, used_llm=True)
        except llm.LLMError as exc:
            return Answer(llm.fallback_answer(hits), sources, used_llm=False, warning=str(exc))
