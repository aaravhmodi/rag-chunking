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
            diagnostics_output = root / "diagnostics.csv"
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
                "--diagnostics-output",
                str(diagnostics_output),
                "--report-output",
                str(report_output),
                "--plots-dir",
                str(plots_dir),
            ]

            with patch("sys.argv", argv):
                main()

            payload = json.loads(json_output.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["overall"]), 2)
            self.assertIn("fixed-128", payload["grouped"])
            self.assertIn("dataset=sample", payload["grouped"]["fixed-128"])
            self.assertTrue(csv_output.exists())
            self.assertTrue(diagnostics_output.exists())
            report_text = report_output.read_text(encoding="utf-8")
            self.assertIn("# RAG Chunking Benchmark Report", report_text)
            self.assertIn("All tables, figures, and summary statements are computed directly from the experiment runs", report_text)
            self.assertIn("## Experimental Results", report_text)
            self.assertIn("## Slice Analysis", report_text)
            self.assertIn("## Diagnostics", report_text)
            self.assertTrue((plots_dir / "quality.jpg").exists())
            self.assertTrue((plots_dir / "latency.jpg").exists())


if __name__ == "__main__":
    unittest.main()
