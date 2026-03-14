import numpy as np


class SimpleVectorIndex:
    def __init__(self, vectors=None, ids=None):
        self.vectors = None
        self.ids = []
        if vectors is not None and ids is not None:
            self.build(vectors, ids)

    def build(self, vectors, ids):
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError("vectors must be a 2D array")
        if len(ids) != len(vectors):
            raise ValueError("ids and vectors must have the same length")

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        self.vectors = vectors / norms
        self.ids = list(ids)
        return self

    def search(self, query_vectors, topk):
        if self.vectors is None:
            raise ValueError("index has not been built")

        query_vectors = np.asarray(query_vectors, dtype=np.float32)
        if query_vectors.ndim == 1:
            query_vectors = query_vectors[None, :]

        norms = np.linalg.norm(query_vectors, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        query_vectors = query_vectors / norms

        scores = query_vectors @ self.vectors.T
        order = np.argsort(-scores, axis=1)[:, :topk]
        top_scores = np.take_along_axis(scores, order, axis=1)
        top_ids = [[self.ids[idx] for idx in row] for row in order]
        return top_scores, top_ids
