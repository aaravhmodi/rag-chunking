from __future__ import annotations

from collections import Counter
from heapq import nlargest
import math

from rag_chunking.models import Chunk, RetrievalResult
from rag_chunking.text_utils import token_counts


class LexicalRetriever:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        self.chunk_vectors: dict[str, Counter[str]] = {}
        self.chunk_norms: dict[str, float] = {}
        self.inverted_index: dict[str, list[tuple[str, int]]] = {}

        for chunk in chunks:
            vector = token_counts(chunk.text)
            self.chunk_vectors[chunk.chunk_id] = vector
            self.chunk_norms[chunk.chunk_id] = math.sqrt(sum(value * value for value in vector.values()))
            for token, frequency in vector.items():
                self.inverted_index.setdefault(token, []).append((chunk.chunk_id, frequency))

    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        query_vector = token_counts(query)
        if not query_vector or top_k <= 0:
            return []

        query_norm = math.sqrt(sum(value * value for value in query_vector.values()))
        if not query_norm:
            return []

        dot_products: dict[str, float] = {}
        for token, query_frequency in query_vector.items():
            for chunk_id, chunk_frequency in self.inverted_index.get(token, []):
                dot_products[chunk_id] = dot_products.get(chunk_id, 0.0) + (query_frequency * chunk_frequency)

        scored = [
            RetrievalResult(
                chunk=self.chunk_by_id[chunk_id],
                score=dot_product / (query_norm * self.chunk_norms[chunk_id]),
            )
            for chunk_id, dot_product in dot_products.items()
            if self.chunk_norms[chunk_id]
        ]
        return nlargest(top_k, scored, key=lambda result: result.score)
