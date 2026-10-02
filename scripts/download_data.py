"""Download the small public RgDAAO source files from Zenodo.

This intentionally skips the multi-GB raw sequencing archives. The benchmark
starts from the authors' deposited processed resources.
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.request import urlopen, urlretrieve

RECORD_API = "https://zenodo.org/api/records/8388902"
KEEP = {
    "DAOx_ref.fa",
    "Figures_data.zip",
    "Jupyter_notebooks_fitness_calculation.zip",
    "Look_up_table.tsv",
}


def main() -> None:
    destination = Path("data/source")
    destination.mkdir(parents=True, exist_ok=True)

    with urlopen(RECORD_API) as response:
        record = json.load(response)

    manifest = []
    for item in record["files"]:
        name = item["key"]
        if name not in KEEP:
            continue
        target = destination / name
        if not target.exists():
            print(f"Downloading {name}...")
            urlretrieve(item["links"]["self"], target)
        manifest.append({
            "name": name,
            "bytes": item["size"],
            "checksum": item.get("checksum"),
            "zenodo_record": "8388902",
        })

    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"Prepared {len(manifest)} source files in {destination}")


if __name__ == "__main__":
    main()
