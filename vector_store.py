"""A small vector database. Uses FAISS when installed, plain NumPy otherwise."""

from __future__ import annotations

import numpy as np

from text_chunker import Chunk

try:
    import faiss  # type: ignore

    faiss.omp_set_num_threads(1)  # avoids thread clashes with PyTorch
    HAS_FAISS = True
except ImportError:  # keeps the project usable on machines where FAISS fails to install
    HAS_FAISS = False


class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.chunks: list[Chunk] = []
        if HAS_FAISS:
            # Vectors are normalized, so inner product = cosine similarity.
            self._index = faiss.IndexFlatIP(dim)
        else:
            self._matrix = np.empty((0, dim), dtype="float32")

    def __len__(self) -> int:
        return len(self.chunks)

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        if not chunks:
            return
        vectors = np.ascontiguousarray(vectors, dtype="float32")
        if HAS_FAISS:
            self._index.add(vectors)
        else:
            self._matrix = np.vstack([self._matrix, vectors])
        self.chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, k: int = 4) -> list[tuple[Chunk, float]]:
        if not self.chunks:
            return []
        k = min(k, len(self.chunks))
        query = np.ascontiguousarray(query_vector.reshape(1, -1), dtype="float32")

        if HAS_FAISS:
            scores, ids = self._index.search(query, k)
            pairs = zip(ids[0], scores[0])
        else:
            all_scores = (self._matrix @ query.T).ravel()
            top = np.argsort(-all_scores)[:k]
            pairs = zip(top, all_scores[top])

        return [(self.chunks[int(i)], float(s)) for i, s in pairs if i != -1]

    def remove_source(self, source: str) -> None:
        """Delete every chunk of one file and rebuild the index."""
        keep = [i for i, c in enumerate(self.chunks) if c.source != source]
        if len(keep) == len(self.chunks):
            return
        if HAS_FAISS:
            vectors = np.vstack([self._index.reconstruct(i) for i in keep]) if keep else None
            self._index = faiss.IndexFlatIP(self.dim)
            if vectors is not None:
                self._index.add(vectors.astype("float32"))
        else:
            self._matrix = self._matrix[keep]
        self.chunks = [self.chunks[i] for i in keep]
