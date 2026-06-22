import unittest

from tests import _path  # noqa: F401

from rag_chunking.chunkers import adaptive_chunks, fixed_token_chunks, paragraph_chunks, sentence_chunks
from rag_chunking.models import Document


class ChunkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.document = Document(
            doc_id="doc1",
            text=(
                "Alpha beta gamma. Delta epsilon zeta.\n\n"
                "Section 2 lists 1970, 1980, 1990 and Toronto Canada references.\n\n"
                "This is a longer explanatory paragraph that keeps discussing the same concept in a more narrative form."
            ),
        )

    def test_fixed_chunks_return_data(self) -> None:
        chunks = fixed_token_chunks(self.document, chunk_size=4)
        self.assertGreaterEqual(len(chunks), 2)

    def test_sentence_chunks_preserve_boundaries(self) -> None:
        chunks = sentence_chunks(self.document, target_tokens=5)
        self.assertTrue(all(chunk.text.endswith((".", "references.", "form.")) for chunk in chunks))

    def test_paragraph_chunks_group_text(self) -> None:
        chunks = paragraph_chunks(self.document, max_tokens=20)
        self.assertGreaterEqual(len(chunks), 2)

    def test_adaptive_chunks_return_strategy_name(self) -> None:
        chunks = adaptive_chunks(self.document)
        self.assertTrue(chunks)
        self.assertTrue(all(chunk.strategy == "adaptive" for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
