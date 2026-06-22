from __future__ import annotations

import json
from pathlib import Path

from rag_chunking.models import Document, QuestionExample


def load_documents(documents_dir: str | Path) -> list[Document]:
    root = Path(documents_dir)
    documents: list[Document] = []
    for path in sorted(root.glob("*.txt")):
        documents.append(Document(doc_id=path.stem, text=path.read_text(encoding="utf-8")))
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
                question=row["question"],
                answer=row["answer"],
                source_doc=row["source_doc"],
                gold_evidence=row["gold_evidence"],
                question_type=row.get("question_type", "unknown"),
                metadata=row.get("metadata", {}),
            )
        )
    return questions
