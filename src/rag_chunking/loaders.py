from __future__ import annotations

import json
from pathlib import Path

from rag_chunking.models import Document, QuestionExample


def load_documents(documents_dir: str | Path) -> list[Document]:
    root = Path(documents_dir)
    documents: list[Document] = []
    for path in sorted(root.glob("*.txt")):
        documents.append(Document(doc_id=path.stem.replace("__", ":"), text=path.read_text(encoding="utf-8")))
    return documents


def load_questions(questions_path: str | Path) -> list[QuestionExample]:
    path = Path(questions_path)
    questions: list[QuestionExample] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        questions.append(
            QuestionExample(
                question_id=row.get("question_id", f"q{len(questions)}"),
                question=row["question"],
                answer=row.get("answer", ""),
                source_doc=row.get("source_doc", ""),
                gold_evidence=row.get("gold_evidence", ""),
                alternative_answers=row.get("alternative_answers", []),
                relevant_doc_ids=row.get("relevant_doc_ids", []),
                evidence_start=row.get("evidence_start"),
                evidence_end=row.get("evidence_end"),
                question_type=row.get("question_type", "unknown"),
                difficulty=row.get("difficulty", "unknown"),
                metadata=row.get("metadata", {}),
            )
        )
    return questions
