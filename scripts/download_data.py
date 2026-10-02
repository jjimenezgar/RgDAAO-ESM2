"""Download only processed data; verify every byte before accepting it."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

# Pinned Zenodo record 8388902; never use the mutable latest-version alias.
FILES = {
    'DAOx_ref.fa': (2113, '7078bab65f38ed6491b31d2552d99bc0'),
    'Figures_data.zip': (4301939, '4129657522b020f6e9160a5b666acd51'),
    'Jupyter_notebooks_fitness_calculation.zip': (14081067, '6bdcad690b3feb594f36136abac99a11'),
    'Look_up_table.tsv': (6709459, 'c15c42fbd94b280f58d415a090a600a2'),
}


def matches(path: Path, size: int, checksum: str) -> bool:
    return (path.is_file() and path.stat().st_size == size
            and hashlib.md5(path.read_bytes()).hexdigest() == checksum)


def download(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name, (size, checksum) in FILES.items():
        target = destination / name
        url = f'https://zenodo.org/records/8388902/files/{name}?download=1'
        if not matches(target, size, checksum):
            temporary = target.with_suffix(target.suffix + '.part')
            for attempt in range(3):
                try:
                    with urlopen(url, timeout=120) as response, temporary.open('wb') as out:
                        while block := response.read(1024 * 1024):
                            out.write(block)
                    if not matches(temporary, size, checksum):
                        raise ValueError(f'Size/checksum mismatch: {name}')
                    temporary.replace(target)
                    break
                except (OSError, ValueError):
                    temporary.unlink(missing_ok=True)
                    if attempt == 2:
                        raise
        manifest.append({'name': name, 'bytes': size, 'checksum': f'md5:{checksum}',
                         'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                         'url': url, 'zenodo_record': '8388902'})
        print(f'Verified {name} ({size:,} bytes)')
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('data/source'))
    download(parser.parse_args().output)
