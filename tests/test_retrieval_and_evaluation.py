import unittest

from tests import _path  # noqa: F401

from rag_chunking.evaluation import answer_exact_match, evidence_span_recall_at_k, ndcg_at_k, recall_at_k, reciprocal_rank
from rag_chunking.models import Chunk, QuestionExample, RetrievalResult
from rag_chunking.retrieval import LexicalRetriever


class RetrievalEvaluationTests(unittest.TestCase):
    def test_retrieval_and_metrics_identify_relevant_chunk(self) -> None:
        chunks = [
            Chunk(
                chunk_id="doc1:sentence:0",
                doc_id="doc1",
                text="Members of the FLQ kidnapped British diplomat James Cross.",
                start_char=0,
                end_char=60,
                strategy="sentence",
            ),
            Chunk(
                chunk_id="doc2:sentence:0",
                doc_id="doc2",
                text="Tokenization speed depends on implementation details.",
                start_char=0,
                end_char=55,
                strategy="sentence",
            ),
        ]
        question = QuestionExample(
            question_id="q1",
            question="Which group kidnapped James Cross?",
            answer="FLQ",
            source_doc="doc1",
            gold_evidence="FLQ kidnapped British diplomat James Cross",
        )

        results = LexicalRetriever(chunks).retrieve(question.question, top_k=2)

        self.assertEqual(results[0].chunk.doc_id, "doc1")
        self.assertEqual(len(results), 1)
        self.assertGreater(results[0].score, 0.0)
        self.assertEqual(recall_at_k(results, question), 1.0)
        self.assertEqual(reciprocal_rank(results, question), 1.0)
        self.assertEqual(ndcg_at_k(results, question), 1.0)
        self.assertEqual(answer_exact_match(results, question), 1.0)

    def test_evidence_span_recall_uses_chunk_boundaries(self) -> None:
        chunks = [
            Chunk(
                chunk_id="doc1:paragraph:0",
                doc_id="doc1",
                text="Alpha beta gamma delta epsilon zeta",
                start_char=0,
                end_char=35,
                strategy="paragraph",
            ),
            Chunk(
                chunk_id="doc1:paragraph:1",
                doc_id="doc1",
                text="eta theta iota kappa lambda",
                start_char=36,
                end_char=63,
                strategy="paragraph",
            ),
        ]
        question = QuestionExample(
            question_id="q2",
            question="Where is gamma delta epsilon discussed?",
            source_doc="doc1",
            gold_evidence="gamma delta epsilon",
            evidence_start=11,
            evidence_end=30,
        )

        results = [RetrievalResult(chunk=chunks[0], score=1.0), RetrievalResult(chunk=chunks[1], score=0.5)]

        self.assertEqual(evidence_span_recall_at_k(results, question), 1.0)


if __name__ == "__main__":
    unittest.main()
