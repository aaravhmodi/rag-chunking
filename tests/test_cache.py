import tempfile
import unittest
from pathlib import Path

from tests import _path  # noqa: F401

from rag_chunking.models import Document
from rag_chunking.pipeline import chunk_documents


class CacheTests(unittest.TestCase):
    def test_chunk_documents_uses_persisted_cache(self) -> None:
        documents = [
            Document(
                doc_id="doc1",
                text="Alpha beta gamma. Delta epsilon zeta.\n\nA second paragraph appears here.",
            )
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_dir = Path(tmp_dir)
            first_chunks, first_latency = chunk_documents(documents, "paragraph", cache_dir=cache_dir)
            second_chunks, second_latency = chunk_documents(documents, "paragraph", cache_dir=cache_dir)

        self.assertTrue(first_chunks)
        self.assertEqual([chunk.text for chunk in first_chunks], [chunk.text for chunk in second_chunks])
        self.assertGreaterEqual(first_latency, 0.0)
        self.assertEqual(second_latency, 0.0)


if __name__ == "__main__":
    unittest.main()
