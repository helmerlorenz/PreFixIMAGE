"""Offline tests: first-run external URLs are not available in this execution sandbox."""
import io
from pathlib import Path
import nbformat
import pytest

from app import online


def test_download_uses_existing_cache_without_network(tmp_path, monkeypatch):
    p = tmp_path / 'existing.py'
    p.write_bytes(b'# upstream source\n' + b'x' * 200)
    monkeypatch.setattr(online, 'urlopen', lambda *a, **kw: (_ for _ in ()).throw(AssertionError('went online')))
    assert online.download('https://example.org/file.py', p, min_bytes=100) == p


def test_download_failure_does_not_leave_partial_file(tmp_path, monkeypatch):
    def no_internet(*a, **kw):
        raise OSError('no network')
    monkeypatch.setattr(online, 'urlopen', no_internet)
    with pytest.raises(OSError):
        online.download('https://example.org/file.py', tmp_path/'new.py')
    assert not (tmp_path/'new.py').exists()
    assert not list(tmp_path.glob('partial_*'))


def test_notebooks_have_no_code_syntax_errors():
    folder = Path(__file__).resolve().parents[1] / 'notebooks'
    assert len(list(folder.glob('*.ipynb'))) == 3
    for f in folder.glob('*.ipynb'):
        notebook = nbformat.read(f, as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type == 'code':
                compile(cell.source, str(f), 'exec')
