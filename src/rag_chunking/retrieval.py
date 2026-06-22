from __future__ import annotations

from collections import Counter

from rag_chunking.models import Chunk, RetrievalResult
from rag_chunking.text_utils import cosine_similarity, token_counts


class LexicalRetriever:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.chunk_vectors: dict[str, Counter[str]] = {chunk.chunk_id: token_counts(chunk.text) for chunk in chunks}

    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        query_vector = token_counts(query)
        scored = [
            RetrievalResult(chunk=chunk, score=cosine_similarity(query_vector, self.chunk_vectors[chunk.chunk_id]))
            for chunk in self.chunks
        ]
        scored.sort(key=lambda result: result.score, reverse=True)
        return scored[:top_k]
