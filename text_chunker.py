"""Split page text into small overlapping chunks.

Why chunk? An embedding model works best on short pieces of text, and we
only want to send the most useful pieces to the LLM, not the whole book.
"""

from __future__ import annotations

from dataclasses import dataclass

from pdf_loader import PageText


@dataclass
class Chunk:
    text: str
    source: str
    page: int


def split_words(text: str, chunk_size: int = 180, overlap: int = 40) -> list[str]:
    """Cut text into pieces of about `chunk_size` words.

    Each piece repeats the last `overlap` words of the previous one, so an
    idea that sits on the border of two chunks is not lost.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be bigger than 0")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be 0 or more, and smaller than chunk_size")

    words = text.split()
    if not words:
        return []

    step = chunk_size - overlap
    pieces = []
    for start in range(0, len(words), step):
        piece = words[start:start + chunk_size]
        pieces.append(" ".join(piece))
        if start + chunk_size >= len(words):
            break
    return pieces


def chunk_pages(pages: list[PageText], chunk_size: int = 180, overlap: int = 40) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in pages:
        for piece in split_words(page.text, chunk_size, overlap):
            chunks.append(Chunk(text=piece, source=page.source, page=page.page))
    return chunks
