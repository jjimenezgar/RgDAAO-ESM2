from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import load_variants
from .split import random_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and split the RgDAAO variant dataset.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = load_variants(args.input)
    train, val, test = random_split(df, seed=args.seed)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    train.to_csv(out / "train.csv", index=False)
    val.to_csv(out / "val.csv", index=False)
    test.to_csv(out / "test.csv", index=False)

    summary = {
        "n_total": len(df),
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "seed": args.seed,
    }
    (out / "split_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
