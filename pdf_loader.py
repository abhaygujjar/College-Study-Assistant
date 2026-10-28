"""Read text out of the files a student uploads (PDF, TXT, MD)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pymupdf  # PyMuPDF


@dataclass
class PageText:
    """Text of one page, with where it came from."""

    source: str
    page: int
    text: str


def _clean(text: str) -> str:
    # PDFs often have broken lines and extra spaces. Make them tidy.
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def read_pdf(data: bytes, filename: str) -> list[PageText]:
    pages: list[PageText] = []
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        for number, page in enumerate(doc, start=1):
            text = _clean(page.get_text("text"))
            if text:
                pages.append(PageText(filename, number, text))
    return pages


def read_text_file(data: bytes, filename: str) -> list[PageText]:
    text = _clean(data.decode("utf-8", errors="ignore"))
    return [PageText(filename, 1, text)] if text else []


def load_document(data: bytes, filename: str) -> list[PageText]:
    """Pick the right reader using the file extension."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        return read_pdf(data, filename)
    if suffix in {".txt", ".md"}:
        return read_text_file(data, filename)
    raise ValueError(f"Unsupported file type: {suffix or 'unknown'}")
