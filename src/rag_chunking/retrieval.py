from __future__ import annotations

from collections import Counter
from heapq import nlargest
import math
from typing import Protocol

from rag_chunking.embeddings import EmbeddingService, build_embedding_service, cosine_similarity
from rag_chunking.models import Chunk, RetrievalResult
from rag_chunking.text_utils import tokenize


class Retriever(Protocol):
    name: str

    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        ...


class LexicalRetriever:
    name = "lexical"

    def __init__(self, chunks: list[Chunk], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        self.chunk_lengths: dict[str, int] = {}
        self.document_frequencies: Counter[str] = Counter()
        self.inverted_index: dict[str, list[tuple[str, int]]] = {}
        self.total_chunks = len(chunks)
        self.average_chunk_length = 0.0

        total_length = 0
        for chunk in chunks:
            vector = Counter(tokenize(chunk.text))
            self.chunk_lengths[chunk.chunk_id] = sum(vector.values())
            total_length += self.chunk_lengths[chunk.chunk_id]
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

        scored = [RetrievalResult(chunk=self.chunk_by_id[chunk_id], score=score, backend=self.name) for chunk_id, score in scores.items()]
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


class EmbeddingRetriever:
    def __init__(self, chunks: list[Chunk], embedding_service: EmbeddingService) -> None:
        self.chunks = chunks
        self.embedding_service = embedding_service
        self.name = f"embedding:{embedding_service.name}"
        self.chunk_embeddings = embedding_service.embed_texts([chunk.text for chunk in chunks])

    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        if top_k <= 0 or not self.chunks:
            return []
        query_vector = self.embedding_service.embed_texts([query])[0]
        scored = [
            RetrievalResult(
                chunk=chunk,
                score=cosine_similarity(query_vector, embedding),
                backend=self.name,
            )
            for chunk, embedding in zip(self.chunks, self.chunk_embeddings)
        ]
        return nlargest(top_k, scored, key=lambda result: result.score)


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], embedding_service: EmbeddingService, lexical_weight: float = 0.5) -> None:
        self.lexical = LexicalRetriever(chunks)
        self.semantic = EmbeddingRetriever(chunks, embedding_service)
        self.lexical_weight = lexical_weight
        self.semantic_weight = 1.0 - lexical_weight
        self.name = f"hybrid:{embedding_service.name}"

    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        if top_k <= 0:
            return []
        lexical_results = self.lexical.retrieve(query, top_k=max(top_k * 4, top_k))
        semantic_results = self.semantic.retrieve(query, top_k=max(top_k * 4, top_k))
        scores: dict[str, float] = {}
        chunk_by_id: dict[str, Chunk] = {}

        lexical_max = max((result.score for result in lexical_results), default=1.0) or 1.0
        semantic_max = max((result.score for result in semantic_results), default=1.0) or 1.0

        for result in lexical_results:
            scores[result.chunk.chunk_id] = scores.get(result.chunk.chunk_id, 0.0) + self.lexical_weight * (result.score / lexical_max)
            chunk_by_id[result.chunk.chunk_id] = result.chunk
        for result in semantic_results:
            scores[result.chunk.chunk_id] = scores.get(result.chunk.chunk_id, 0.0) + self.semantic_weight * (result.score / semantic_max)
            chunk_by_id[result.chunk.chunk_id] = result.chunk

        combined = [RetrievalResult(chunk=chunk_by_id[chunk_id], score=score, backend=self.name) for chunk_id, score in scores.items()]
        return nlargest(top_k, combined, key=lambda result: result.score)


def build_retriever(
    chunks: list[Chunk],
    retriever_spec: str = "lexical",
    embedding_backend: str | None = None,
) -> Retriever:
    normalized = retriever_spec.strip().lower()
    if normalized == "lexical":
        return LexicalRetriever(chunks)
    if normalized == "embedding":
        return EmbeddingRetriever(chunks, build_embedding_service(embedding_backend))
    if normalized == "hybrid":
        return HybridRetriever(chunks, build_embedding_service(embedding_backend))
    raise ValueError(f"Unsupported retriever backend: {retriever_spec}")
