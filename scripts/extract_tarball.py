from __future__ import annotations

import sys
import tarfile
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("Usage: extract_tarball.py <archive> <target-dir>")

    archive = Path(sys.argv[1])
    target_dir = Path(sys.argv[2])
    target_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(target_dir)

    print(f"extracted {archive} -> {target_dir}")


if __name__ == "__main__":
    main()
