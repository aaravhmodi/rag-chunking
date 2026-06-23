from __future__ import annotations

from collections import Counter
from heapq import nlargest
import math

from rag_chunking.models import Chunk, RetrievalResult
from rag_chunking.text_utils import tokenize


class LexicalRetriever:
    def __init__(self, chunks: list[Chunk], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        self.chunk_term_frequencies: dict[str, Counter[str]] = {}
        self.chunk_lengths: dict[str, int] = {}
        self.document_frequencies: Counter[str] = Counter()
        self.inverted_index: dict[str, list[tuple[str, int]]] = {}
        self.total_chunks = len(chunks)
        self.average_chunk_length = 0.0

        total_length = 0
        for chunk in chunks:
            tokens = tokenize(chunk.text)
            vector = Counter(tokens)
            self.chunk_term_frequencies[chunk.chunk_id] = vector
            self.chunk_lengths[chunk.chunk_id] = len(tokens)
            total_length += len(tokens)
            for token, frequency in vector.items():
                self.inverted_index.setdefault(token, []).append((chunk.chunk_id, frequency))
            self.document_frequencies.update(vector.keys())
        self.average_chunk_length = total_length / self.total_chunks if self.total_chunks else 0.0

    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        query_vector = Counter(tokenize(query))
        if not query_vector or top_k <= 0:
            return []

        scores: dict[str, float] = {}
        for token, query_frequency in query_vector.items():
            idf = self._idf(token)
            if idf <= 0:
                continue
            for chunk_id, chunk_frequency in self.inverted_index.get(token, []):
                scores[chunk_id] = scores.get(chunk_id, 0.0) + (query_frequency * idf * self._bm25_tf(chunk_id, chunk_frequency))

        scored = [
            RetrievalResult(
                chunk=self.chunk_by_id[chunk_id],
                score=score,
            )
            for chunk_id, score in scores.items()
        ]
        return nlargest(top_k, scored, key=lambda result: result.score)

    def _idf(self, token: str) -> float:
        document_frequency = self.document_frequencies.get(token, 0)
        if document_frequency == 0 or self.total_chunks == 0:
            return 0.0
        numerator = self.total_chunks - document_frequency + 0.5
        denominator = document_frequency + 0.5
        return math.log1p(numerator / denominator)

    def _bm25_tf(self, chunk_id: str, chunk_frequency: int) -> float:
        chunk_length = self.chunk_lengths[chunk_id]
        length_norm = 1.0 - self.b + self.b * (chunk_length / self.average_chunk_length) if self.average_chunk_length else 1.0
        denominator = chunk_frequency + self.k1 * length_norm
        if denominator == 0:
            return 0.0
        return (chunk_frequency * (self.k1 + 1.0)) / denominator
