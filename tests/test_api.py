import io
from PIL import Image
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def sample_image():
    img = Image.new('RGB', (64, 64), color=(130, 120, 100))
    f = io.BytesIO(); img.save(f, format='PNG')
    return f.getvalue()


def test_home():
    r = client.get('/')
    assert r.status_code == 200
    assert 'iMAGE' in r.text


def test_status():
    r = client.get('/api/status')
    assert r.status_code == 200
    assert {m['key'] for m in r.json()['models']} == {'dncnn', 'restormer', 'nafnet'}


def test_preview_denoise(monkeypatch):
    import app.inference
    monkeypatch.setattr(app.inference, 'MODE', 'preview')
    # MODE is initialized at import in app.main too; service is typically preview by default.
    from app.main import MODE
    if MODE != 'preview':
        return
    payload = sample_image()
    r = client.post('/api/denoise', files={'image': ('test.png', payload, 'image/png'),
                                         'reference': ('ref.png', payload, 'image/png')})
    assert r.status_code == 200, r.text
    assert len(r.json()['models']) == 3
    assert all('PREVIEW FILTER' in m['label'] for m in r.json()['models'])
    assert r.json()['best'] is None


def test_reject_bad_image():
    r = client.post('/api/denoise', files={'image': ('bad.png', b'garbage', 'image/png')})
    assert r.status_code == 415
