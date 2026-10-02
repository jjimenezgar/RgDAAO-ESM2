"""Check corrupted downloads/cache recovery without external network access."""
import hashlib
import importlib.util
import io
from pathlib import Path

spec = importlib.util.spec_from_file_location('data_downloader', Path('scripts/download_data.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_truncated_cache_is_replaced_and_verified(tmp_path, monkeypatch):
    payload = b'verified processed data'
    checksum = hashlib.md5(payload).hexdigest()
    monkeypatch.setattr(module, 'FILES', {'example.dat': (len(payload), checksum)})
    path = tmp_path / 'example.dat'
    path.write_bytes(b'truncated')
    responses = iter([b'bad first transfer', payload])
    monkeypatch.setattr(module, 'urlopen', lambda *a, **k: io.BytesIO(next(responses)))
    module.download(tmp_path)
    assert path.read_bytes() == payload
    assert not (tmp_path / 'example.dat.part').exists()
    # A verified cache hit must not call the network again.
    monkeypatch.setattr(module, 'urlopen', lambda *a, **k: (_ for _ in ()).throw(AssertionError('unexpected network')))
    module.download(tmp_path)
