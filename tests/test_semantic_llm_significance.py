import unittest

from tests import _path  # noqa: F401

from rag_chunking.chunkers import build_chunks
from rag_chunking.llm import build_llm_judge
from rag_chunking.models import Chunk, Document, QuestionExample, RetrievalResult
from rag_chunking.pipeline import run_experiment
from rag_chunking.significance import compare_strategies_by_dataset


class SemanticChunkingAndJudgeTests(unittest.TestCase):
    def test_semantic_chunking_strategy_builds_chunks(self) -> None:
        document = Document(
            doc_id="doc1",
            text="The FLQ kidnapped James Cross. Montreal entered a period of crisis. Tokenization speed depends on implementation details.",
        )

        chunks = build_chunks(document, "semantic", embedding_backend="hash")

        self.assertTrue(chunks)
        self.assertTrue(all(chunk.strategy == "semantic" for chunk in chunks))
        self.assertTrue(all(chunk.metadata.get("embedding_backend") == "hash" for chunk in chunks))

    def test_heuristic_llm_grading_flows_into_results(self) -> None:
        documents = [Document(doc_id="history", text="The FLQ kidnapped James Cross in Montreal during the October Crisis.")]
        questions = [
            QuestionExample(
                question_id="q1",
                question="Which group kidnapped James Cross?",
                answer="FLQ",
                source_doc="history",
                gold_evidence="FLQ kidnapped James Cross",
            )
        ]

        result = run_experiment(documents, questions, strategy="sentence", top_k=3, llm_judge_backend="heuristic")

        self.assertEqual(result.llm_judged_question_count, 1)
        self.assertEqual(result.llm_answer_score, 1.0)
        self.assertEqual(result.hallucination_rate, 0.0)

    def test_heuristic_judge_returns_generated_answer(self) -> None:
        judge = build_llm_judge("heuristic")
        question = QuestionExample(question_id="q1", question="Which group kidnapped James Cross?", answer="FLQ")
        results = [
            RetrievalResult(
                chunk=Chunk(
                    chunk_id="doc1:sentence:0",
                    doc_id="doc1",
                    text="The FLQ kidnapped James Cross in Montreal.",
                    start_char=0,
                    end_char=42,
                    strategy="sentence",
                ),
                score=1.0,
                backend="lexical",
            )
        ]

        grade = judge.grade(question, results)

        self.assertEqual(grade.answer_score, 1.0)
        self.assertEqual(grade.hallucination_score, 0.0)
        self.assertEqual(grade.generated_answer, "FLQ")


class SignificanceTests(unittest.TestCase):
    def test_compare_strategies_by_dataset_returns_comparisons(self) -> None:
        scores = {
            "baseline": {"set1": [0.0, 0.0, 0.0, 0.0]},
            "candidate": {"set1": [1.0, 1.0, 1.0, 1.0]},
        }

        comparisons = compare_strategies_by_dataset("recall_at_k", scores, baseline_strategy="baseline", iterations=200)

        self.assertEqual(len(comparisons), 1)
        self.assertEqual(comparisons[0].dataset, "set1")
        self.assertGreater(comparisons[0].mean_delta, 0.0)


if __name__ == "__main__":
    unittest.main()
