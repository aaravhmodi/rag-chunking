from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from urllib import request

from rag_chunking.text_utils import tokenize


def cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


class EmbeddingBackend:
    name = "base"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class HashEmbeddingBackend(EmbeddingBackend):
    name = "hash"

    def __init__(self, dimensions: int = 128) -> None:
        self.dimensions = dimensions

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = tokenize(text)
        if not tokens:
            return vector
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            slot = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            weight = 1.0 + (digest[5] / 255.0)
            vector[slot] += sign * weight
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]


class SentenceTransformerEmbeddingBackend(EmbeddingBackend):
    name = "sentence-transformers"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("sentence-transformers is not installed") from exc
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [list(vector) for vector in self.model.encode(texts, normalize_embeddings=True)]


class OpenAIEmbeddingBackend(EmbeddingBackend):
    name = "openai"

    def __init__(self, model: str = "text-embedding-3-small", api_key: str | None = None) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        payload = json.dumps({"input": texts, "model": self.model}).encode("utf-8")
        req = request.Request(
            "https://api.openai.com/v1/embeddings",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with request.urlopen(req, timeout=60) as response:  # pragma: no cover - network dependent
            body = json.loads(response.read().decode("utf-8"))
        return [row["embedding"] for row in body["data"]]


@dataclass(slots=True)
class EmbeddingService:
    backend: EmbeddingBackend

    @property
    def name(self) -> str:
        return self.backend.name

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self.backend.embed_texts(texts)


def build_embedding_service(spec: str | None) -> EmbeddingService:
    normalized = (spec or "hash").strip().lower()
    if normalized in {"hash", "local"}:
        return EmbeddingService(HashEmbeddingBackend())
    if normalized.startswith("sentence-transformers"):
        parts = normalized.split(":", maxsplit=1)
        model_name = parts[1] if len(parts) == 2 and parts[1] else "all-MiniLM-L6-v2"
        return EmbeddingService(SentenceTransformerEmbeddingBackend(model_name=model_name))
    if normalized.startswith("openai"):
        parts = normalized.split(":", maxsplit=1)
        model_name = parts[1] if len(parts) == 2 and parts[1] else "text-embedding-3-small"
        return EmbeddingService(OpenAIEmbeddingBackend(model=model_name))
    raise ValueError(f"Unsupported embedding backend: {spec}")
