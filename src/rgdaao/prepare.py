from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path

from .data import load_variants
from .split import random_split
from .source import read_fasta


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and split the RgDAAO variant dataset.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    reference = Path(args.input).parent / "wt.fasta"
    df = load_variants(args.input, read_fasta(reference) if reference.exists() else None)
    train, val, test = random_split(df, seed=args.seed)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    distributions = {}
    hashes = {}
    for name, frame in (("train", train), ("val", val), ("test", test)):
        path = out / f"{name}.csv"
        frame.to_csv(path, index=False, float_format="%.17g", lineterminator="\n")
        distributions[name] = frame.activity.describe().to_dict()
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()

    summary = {
        "n_total": len(df),
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "seed": args.seed,
        "fractions": [0.8, 0.1, 0.1],
        "target_distributions": distributions,
        "split_sha256": hashes,
        "dataset_sha256": hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
    }
    (out / "split_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
