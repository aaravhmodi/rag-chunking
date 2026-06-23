import json
import tempfile
import unittest
from pathlib import Path

from tests import _path  # noqa: F401

from rag_chunking.loaders import load_documents, load_questions


class LoaderTests(unittest.TestCase):
    def test_load_documents_and_questions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            documents_dir = root / "documents"
            documents_dir.mkdir()
            (documents_dir / "alpha.txt").write_text("Alpha text", encoding="utf-8")
            (documents_dir / "beta.txt").write_text("Beta text", encoding="utf-8")

            questions_path = root / "questions.jsonl"
            rows = [
                {
                    "question_id": "alpha-q1",
                    "question": "What is alpha?",
                    "answer": "Alpha",
                    "source_doc": "alpha",
                    "gold_evidence": "Alpha text",
                    "question_type": "definition",
                    "metadata": {"difficulty": "easy"},
                }
            ]
            questions_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

            documents = load_documents(documents_dir)
            questions = load_questions(questions_path)

        self.assertEqual([document.doc_id for document in documents], ["alpha", "beta"])
        self.assertEqual(len(questions), 1)
        self.assertEqual(questions[0].question_id, "alpha-q1")
        self.assertEqual(questions[0].metadata["difficulty"], "easy")


if __name__ == "__main__":
    unittest.main()
