import unittest

from tests import _path  # noqa: F401

from rag_chunking.models import Document, QuestionExample
from rag_chunking.pipeline import run_experiment


class PipelineTests(unittest.TestCase):
    def test_run_experiment_returns_metrics(self) -> None:
        documents = [
            Document(
                doc_id="history",
                text="The FLQ kidnapped James Cross in Montreal during the October Crisis.",
            )
        ]
        questions = [
            QuestionExample(
                question="Which group kidnapped James Cross?",
                answer="FLQ",
                source_doc="history",
                gold_evidence="FLQ kidnapped James Cross",
            )
        ]

        result = run_experiment(documents, questions, strategy="sentence", top_k=3)

        self.assertEqual(result.strategy, "sentence")
        self.assertEqual(result.top_k, 3)
        self.assertGreaterEqual(result.recall_at_k, 0.0)
        self.assertLessEqual(result.recall_at_k, 1.0)
        self.assertGreater(result.avg_chunk_count, 0.0)
        self.assertGreater(result.avg_chunk_length_chars, 0.0)


if __name__ == "__main__":
    unittest.main()
