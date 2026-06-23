import unittest

from tests import _path  # noqa: F401

from rag_chunking.models import Document, QuestionExample
from rag_chunking.pipeline import run_experiment, run_grouped_experiments


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
                question_id="q1",
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
        self.assertEqual(result.answerable_question_count, 1)
        self.assertEqual(result.total_question_count, 1)
        self.assertGreater(result.avg_chunk_count, 0.0)
        self.assertGreater(result.avg_chunk_length_chars, 0.0)

    def test_run_grouped_experiments_slices_by_metadata(self) -> None:
        documents = [
            Document(doc_id="history", text="The FLQ kidnapped James Cross in Montreal during the October Crisis."),
            Document(doc_id="science", text="Seed lexicons contain positive and negative predicates for event polarity."),
        ]
        questions = [
            QuestionExample(
                question_id="q1",
                question="Which group kidnapped James Cross?",
                answer="FLQ",
                source_doc="history",
                gold_evidence="FLQ kidnapped James Cross",
                metadata={"dataset": "hist", "split": "test"},
            ),
            QuestionExample(
                question_id="q2",
                question="What does the seed lexicon contain?",
                answer="positive and negative predicates",
                source_doc="science",
                gold_evidence="positive and negative predicates",
                metadata={"dataset": "science", "split": "dev"},
            ),
        ]

        grouped = run_grouped_experiments(documents, questions, strategy="sentence", top_k=3)

        self.assertIn("overall", grouped)
        self.assertIn("dataset=hist", grouped)
        self.assertIn("dataset=science", grouped)
        self.assertIn("split=test", grouped)
        self.assertEqual(grouped["dataset=hist"].total_question_count, 1)
        self.assertEqual(grouped["split=dev"].total_question_count, 1)


if __name__ == "__main__":
    unittest.main()
