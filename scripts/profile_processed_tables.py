"""Inspect workbook schemas and notebook inputs without executing notebooks."""
import io
import json
import zipfile
from pathlib import Path
import pandas as pd


def main():
    for path in sorted(Path('data/source').glob('*.zip')):
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if name.endswith('.xlsx'):
                    book = pd.ExcelFile(io.BytesIO(archive.read(name)))
                    for sheet in book.sheet_names:
                        frame = book.parse(sheet)
                        print(name, sheet, frame.shape, frame.columns.tolist())
                elif name.endswith('.csv'):
                    with archive.open(name) as stream:
                        frame = pd.read_csv(stream, sep='\t', nrows=3)
                    print(name, 'tab-separated input columns:', frame.columns.tolist())
                elif name.endswith('.ipynb'):
                    notebook = json.loads(archive.read(name))
                    print(name, 'cells:', len(notebook['cells']))
                    for i, cell in enumerate(notebook['cells']):
                        for line in ''.join(cell.get('source', [])).splitlines():
                            if any(t in line for t in ('logfitness', 'min_max_norm(np', "mgd['activity']", "mgd['expression']", ' - 104')):
                                print(f'  cell {i}: {line}')


if __name__ == '__main__':
    main()
