"""Inspect Zenodo record 8388902 without downloading multi-GB sequencing archives.

Usage:
    python scripts/inspect_zenodo.py

This script prints filenames, sizes and direct download links from the public
Zenodo API. It is intentionally an inspection step: the benchmark should use
the smallest processed table that faithfully contains mutation identities and
activity fitness, rather than reprocessing raw sequencing unnecessarily.
"""

from __future__ import annotations

import json
from urllib.request import urlopen

RECORD = "https://zenodo.org/api/records/8388902"


def main() -> None:
    with urlopen(RECORD) as response:
        record = json.load(response)

    for item in record["files"]:
        print(f'{item["key"]}\t{item["size"]}\t{item["links"]["self"]}')


if __name__ == "__main__":
    main()
