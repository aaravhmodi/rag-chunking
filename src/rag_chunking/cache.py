from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from rag_chunking.models import Chunk, Document


def load_cached_chunks(cache_dir: str | Path, documents: list[Document], strategy: str) -> list[Chunk] | None:
    cache_path = _cache_path(cache_dir, documents, strategy)
    if not cache_path.exists():
        return None
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    return [Chunk(**row) for row in payload["chunks"]]


def save_cached_chunks(cache_dir: str | Path, documents: list[Document], strategy: str, chunks: list[Chunk]) -> Path:
    cache_path = _cache_path(cache_dir, documents, strategy)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "strategy": strategy,
        "document_count": len(documents),
        "chunks": [asdict(chunk) for chunk in chunks],
    }
    cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return cache_path


def _cache_path(cache_dir: str | Path, documents: list[Document], strategy: str) -> Path:
    root = Path(cache_dir)
    cache_key = _document_collection_fingerprint(documents)
    safe_strategy = strategy.replace("/", "_").replace("\\", "_")
    return root / safe_strategy / f"{cache_key}.json"


def _document_collection_fingerprint(documents: list[Document]) -> str:
    digest = hashlib.sha256()
    for document in documents:
        digest.update(document.doc_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(document.text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()[:24]
