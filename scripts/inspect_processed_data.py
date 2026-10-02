"""Inspect deposited processed archives and rank candidate fitness tables."""

from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path

TOKENS = ("fitness", "activity", "expression", "variant", "mutation")


def score(name: str) -> int:
    lower = name.lower()
    return sum(token in lower for token in TOKENS)


def inspect_zip(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        candidates = sorted(archive.namelist(), key=lambda x: (-score(x), x))
        print(f"\n[{path.name}]")
        for name in candidates[:40]:
            if score(name):
                print(name)


def main() -> None:
    root = Path("data/source")
    for path in sorted(root.glob("*.zip")):
        inspect_zip(path)


if __name__ == "__main__":
    main()
