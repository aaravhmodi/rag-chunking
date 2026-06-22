import unittest

from tests import _path  # noqa: F401

from rag_chunking.evaluation import answer_exact_match, ndcg_at_k, recall_at_k, reciprocal_rank
from rag_chunking.models import Chunk, QuestionExample
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
            question="Which group kidnapped James Cross?",
            answer="FLQ",
            source_doc="doc1",
            gold_evidence="FLQ kidnapped British diplomat James Cross",
        )

        results = LexicalRetriever(chunks).retrieve(question.question, top_k=2)

        self.assertEqual(results[0].chunk.doc_id, "doc1")
        self.assertEqual(recall_at_k(results, question), 1.0)
        self.assertEqual(reciprocal_rank(results, question), 1.0)
        self.assertEqual(ndcg_at_k(results, question), 1.0)
        self.assertEqual(answer_exact_match(results, question), 1.0)


if __name__ == "__main__":
    unittest.main()
