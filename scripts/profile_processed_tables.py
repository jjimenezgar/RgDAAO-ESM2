"""Extract and inspect tabular files from the authors' processed deposit.

The script does not guess biological column meanings. It prints table shape,
column names and a small schema summary so the exact activity-fitness source
can be selected explicitly and reproducibly.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

EXTENSIONS = {".csv", ".tsv", ".txt"}


def read_table(path: Path):
    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    return pd.read_csv(path, sep=sep)


def main() -> None:
    source = Path("data/source")
    extracted = source / "extracted"
    extracted.mkdir(parents=True, exist_ok=True)

    for archive_path in source.glob("*.zip"):
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.namelist():
                suffix = Path(member).suffix.lower()
                if suffix not in EXTENSIONS:
                    continue
                target = extracted / archive_path.stem / member
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as src, target.open("wb") as dst:
                    dst.write(src.read())

    candidates = [source / "Look_up_table.tsv"]
    candidates.extend(extracted.rglob("*"))
    for path in candidates:
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            continue
        try:
            df = read_table(path)
        except Exception as exc:
            print(f"SKIP {path}: {exc}")
            continue
        print(f"TABLE {path} rows={len(df)} cols={len(df.columns)}")
        print("COLUMNS:", " | ".join(map(str, df.columns)))
        print()


if __name__ == "__main__":
    main()
