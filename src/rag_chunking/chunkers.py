from __future__ import annotations

import re
from dataclasses import dataclass

from rag_chunking.models import Chunk, Document
from rag_chunking.text_utils import split_paragraphs, split_sentences, tokenize


def _char_span(text: str, fragment: str, offset: int) -> tuple[int, int]:
    start = text.find(fragment, offset)
    if start < 0:
        start = offset
    end = start + len(fragment)
    return start, end


def _make_chunk(
    document: Document,
    text: str,
    start_char: int,
    end_char: int,
    strategy: str,
    index: int,
    metadata: dict | None = None,
) -> Chunk:
    return Chunk(
        chunk_id=f"{document.doc_id}:{strategy}:{index}",
        doc_id=document.doc_id,
        text=text,
        start_char=start_char,
        end_char=end_char,
        strategy=strategy,
        metadata=metadata or {},
    )


def fixed_token_chunks(document: Document, chunk_size: int, overlap: int = 0) -> list[Chunk]:
    words = re.findall(r"\S+", document.text)
    if not words:
        return []
    step = max(1, chunk_size - overlap)
    chunks: list[Chunk] = []
    cursor = 0
    char_offset = 0
    index = 0
    while cursor < len(words):
        window = words[cursor : cursor + chunk_size]
        chunk_text = " ".join(window)
        start_char, end_char = _char_span(document.text, chunk_text, char_offset)
        char_offset = max(char_offset, start_char + 1)
        chunks.append(
            _make_chunk(
                document,
                chunk_text,
                start_char,
                end_char,
                strategy=f"fixed-{chunk_size}",
                index=index,
                metadata={"chunk_size": chunk_size, "overlap": overlap},
            )
        )
        index += 1
        cursor += step
    return chunks


def sentence_chunks(document: Document, target_tokens: int = 256) -> list[Chunk]:
    sentences = split_sentences(document.text)
    chunks: list[Chunk] = []
    current: list[str] = []
    current_tokens = 0
    char_offset = 0
    for sentence in sentences:
        sentence_tokens = len(tokenize(sentence))
        if current and current_tokens + sentence_tokens > target_tokens:
            chunk_text = " ".join(current)
            start_char, end_char = _char_span(document.text, chunk_text, char_offset)
            char_offset = max(char_offset, start_char + 1)
            chunks.append(_make_chunk(document, chunk_text, start_char, end_char, "sentence", len(chunks)))
            current = [sentence]
            current_tokens = sentence_tokens
        else:
            current.append(sentence)
            current_tokens += sentence_tokens
    if current:
        chunk_text = " ".join(current)
        start_char, end_char = _char_span(document.text, chunk_text, char_offset)
        chunks.append(_make_chunk(document, chunk_text, start_char, end_char, "sentence", len(chunks)))
    return chunks


def paragraph_chunks(document: Document, max_tokens: int = 512) -> list[Chunk]:
    paragraphs = split_paragraphs(document.text)
    chunks: list[Chunk] = []
    current: list[str] = []
    current_tokens = 0
    char_offset = 0
    for paragraph in paragraphs:
        paragraph_tokens = len(tokenize(paragraph))
        if current and current_tokens + paragraph_tokens > max_tokens:
            chunk_text = "\n\n".join(current)
            start_char, end_char = _char_span(document.text, chunk_text, char_offset)
            char_offset = max(char_offset, start_char + 1)
            chunks.append(_make_chunk(document, chunk_text, start_char, end_char, "paragraph", len(chunks)))
            current = [paragraph]
            current_tokens = paragraph_tokens
        else:
            current.append(paragraph)
            current_tokens += paragraph_tokens
    if current:
        chunk_text = "\n\n".join(current)
        start_char, end_char = _char_span(document.text, chunk_text, char_offset)
        chunks.append(_make_chunk(document, chunk_text, start_char, end_char, "paragraph", len(chunks)))
    return chunks


def sliding_window_chunks(document: Document, chunk_size: int = 512, stride: int = 256) -> list[Chunk]:
    words = re.findall(r"\S+", document.text)
    if not words:
        return []
    chunks: list[Chunk] = []
    char_offset = 0
    for index, cursor in enumerate(range(0, len(words), stride)):
        chunk_text = " ".join(words[cursor : cursor + chunk_size])
        if not chunk_text:
            continue
        start_char, end_char = _char_span(document.text, chunk_text, char_offset)
        char_offset = max(char_offset, start_char + 1)
        chunks.append(
            _make_chunk(
                document,
                chunk_text,
                start_char,
                end_char,
                "sliding-window",
                index,
                metadata={"chunk_size": chunk_size, "stride": stride},
            )
        )
    return chunks


@dataclass(slots=True)
class ParagraphFeatures:
    token_count: int
    avg_sentence_length: float
    number_count: int
    entity_hint_count: int
    has_list_marker: bool
    has_code_marker: bool


def _paragraph_features(paragraph: str) -> ParagraphFeatures:
    sentences = split_sentences(paragraph)
    sentence_lengths = [len(tokenize(sentence)) for sentence in sentences] or [0]
    tokens = tokenize(paragraph)
    raw_tokens = re.findall(r"\S+", paragraph)
    number_count = sum(token.isdigit() for token in raw_tokens)
    entity_hint_count = sum(token[:1].isupper() for token in raw_tokens if token[:1].isalpha())
    has_list_marker = bool(re.search(r"(^|\n)\s*([-*]|\d+\.)\s+", paragraph))
    has_code_marker = "{" in paragraph or "}" in paragraph or "def " in paragraph or "class " in paragraph
    return ParagraphFeatures(
        token_count=len(tokens),
        avg_sentence_length=sum(sentence_lengths) / len(sentence_lengths),
        number_count=number_count,
        entity_hint_count=entity_hint_count,
        has_list_marker=has_list_marker,
        has_code_marker=has_code_marker,
    )


def adaptive_chunks(document: Document, default_tokens: int = 512) -> list[Chunk]:
    paragraphs = split_paragraphs(document.text)
    chunks: list[Chunk] = []
    current: list[str] = []
    current_budget = default_tokens
    current_tokens = 0
    char_offset = 0

    for paragraph in paragraphs:
        features = _paragraph_features(paragraph)
        if features.has_code_marker or features.has_list_marker:
            target_tokens = 256
        elif features.number_count >= 3 or features.entity_hint_count >= 6:
            target_tokens = 192
        elif features.avg_sentence_length >= 24:
            target_tokens = 768
        elif features.token_count <= 80:
            target_tokens = 256
        else:
            target_tokens = default_tokens

        if current and current_tokens + features.token_count > current_budget:
            chunk_text = "\n\n".join(current)
            start_char, end_char = _char_span(document.text, chunk_text, char_offset)
            char_offset = max(char_offset, start_char + 1)
            chunks.append(
                _make_chunk(
                    document,
                    chunk_text,
                    start_char,
                    end_char,
                    "adaptive",
                    len(chunks),
                    metadata={"target_tokens": current_budget},
                )
            )
            current = [paragraph]
            current_budget = target_tokens
            current_tokens = features.token_count
        else:
            current.append(paragraph)
            current_budget = target_tokens if not current_tokens else max(current_budget, target_tokens)
            current_tokens += features.token_count

    if current:
        chunk_text = "\n\n".join(current)
        start_char, end_char = _char_span(document.text, chunk_text, char_offset)
        chunks.append(
            _make_chunk(
                document,
                chunk_text,
                start_char,
                end_char,
                "adaptive",
                len(chunks),
                metadata={"target_tokens": current_budget},
            )
        )
    return chunks


def build_chunks(document: Document, strategy: str) -> list[Chunk]:
    if strategy.startswith("fixed-"):
        _, size = strategy.split("-", maxsplit=1)
        return fixed_token_chunks(document, chunk_size=int(size))
    if strategy == "sentence":
        return sentence_chunks(document)
    if strategy == "paragraph":
        return paragraph_chunks(document)
    if strategy == "sliding-window":
        return sliding_window_chunks(document)
    if strategy == "adaptive":
        return adaptive_chunks(document)
    raise ValueError(f"Unsupported strategy: {strategy}")
