"""Turn text into vectors (embeddings) using a small model that runs on a laptop."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _load_model():
    # Imported here so the tests do not need PyTorch or the model.
    import torch

    # Streamlit's file watcher looks inside every loaded module. Inside
    # torch.classes this raises an error that can kill the whole server
    # (you see "Connection error" in the browser). An empty path avoids it.
    try:
        torch.classes.__path__ = []
    except Exception:
        pass

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME, device="cpu")
    model.max_seq_length = 256  # our chunks are ~180 words, this keeps memory low
    return model


class ModelLoadError(RuntimeError):
    """The embedding model could not be loaded (usually: not enough memory)."""


def warm_up() -> None:
    """Load the model now, so the first indexing does not do the heavy work."""
    try:
        _load_model()
    except (MemoryError, OSError) as exc:
        out_of_memory = (
            isinstance(exc, MemoryError)
            or getattr(exc, "winerror", None) == 1455
            or "paging file" in str(exc).lower()
        )
        if not out_of_memory:
            raise
        raise ModelLoadError(
            "Windows ran out of memory while loading the embedding model (error 1455). "
            "Close other programs and browser tabs and restart the app. If it keeps "
            "happening, increase the Windows paging file (virtual memory) and restart the PC."
        ) from exc


def embed_texts(texts: list[str]) -> np.ndarray:
    """Return one normalized vector per text, shape (len(texts), 384)."""
    model = _load_model()
    vectors = model.encode(
        texts,
        batch_size=16,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vectors.astype("float32")
