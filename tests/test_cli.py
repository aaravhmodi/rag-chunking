import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests import _path  # noqa: F401

from rag_chunking.cli import main


class CliTests(unittest.TestCase):
    def test_cli_writes_expected_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            json_output = root / "results.json"
            csv_output = root / "results.csv"
            report_output = root / "report.md"
            plots_dir = root / "plots"

            argv = [
                "rag-benchmark",
                "--documents",
                "data/sample/documents",
                "--questions",
                "data/sample/questions.jsonl",
                "--strategies",
                "fixed-128",
                "paragraph",
                "--output",
                str(json_output),
                "--csv-output",
                str(csv_output),
                "--report-output",
                str(report_output),
                "--plots-dir",
                str(plots_dir),
            ]

            with patch("sys.argv", argv):
                main()

            payload = json.loads(json_output.read_text(encoding="utf-8"))
            self.assertEqual(len(payload), 2)
            self.assertTrue(csv_output.exists())
            self.assertIn("# RAG Chunking Benchmark Report", report_output.read_text(encoding="utf-8"))
            self.assertTrue((plots_dir / "quality.svg").exists())
            self.assertTrue((plots_dir / "latency.svg").exists())


if __name__ == "__main__":
    unittest.main()
