from __future__ import annotations

import argparse
import json
import tarfile
import urllib.request
from pathlib import Path

from beir import util


BEIR_BASE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"
QASPER_TRAIN_DEV_URL = "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz"
QASPER_TEST_URL = "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz"


def download_beir_dataset(dataset_name: str, output_root: Path) -> dict[str, str]:
    output_root.mkdir(parents=True, exist_ok=True)
    archive_url = f"{BEIR_BASE_URL}/{dataset_name}.zip"
    data_path = Path(util.download_and_unzip(archive_url, str(output_root)))
    return {
        "dataset": dataset_name,
        "source": "beir",
        "path": str(data_path),
    }


def download_qasper(output_root: Path) -> dict[str, str]:
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": "qasper",
        "splits": {},
    }
    archives = [
        ("train-dev", QASPER_TRAIN_DEV_URL),
        ("test", QASPER_TEST_URL),
    ]
    for archive_name, archive_url in archives:
        archive_path = output_root / f"{archive_name}.tgz"
        urllib.request.urlretrieve(archive_url, archive_path)
        with tarfile.open(archive_path, "r:gz") as handle:
            handle.extractall(output_root)

    split_files = {
        "train": output_root / "qasper-train-v0.3.json",
        "validation": output_root / "qasper-dev-v0.3.json",
        "test": output_root / "qasper-test-v0.3.json",
    }
    for split_name, split_path in split_files.items():
        records = json.loads(split_path.read_text(encoding="utf-8"))
        manifest["splits"][split_name] = {
            "records_path": str(split_path),
            "count": len(records),
        }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "dataset": "qasper",
        "source": "huggingface",
        "path": str(output_root),
        "manifest": str(manifest_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download raw benchmark datasets for the RAG chunking project.")
    parser.add_argument(
        "--beir-datasets",
        nargs="*",
        default=["scifact", "fiqa", "nfcorpus"],
        help="BEIR dataset names to download.",
    )
    parser.add_argument(
        "--output-root",
        default="data/raw",
        help="Root directory for downloaded datasets.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    beir_root = output_root / "beir"
    qasper_root = output_root / "qasper"

    downloads: list[dict[str, str]] = []
    for dataset_name in args.beir_datasets:
        downloads.append(download_beir_dataset(dataset_name, beir_root))
    downloads.append(download_qasper(qasper_root))

    manifest_path = output_root / "download_manifest.json"
    manifest_path.write_text(json.dumps(downloads, indent=2), encoding="utf-8")
    print(json.dumps(downloads, indent=2))


if __name__ == "__main__":
    main()
