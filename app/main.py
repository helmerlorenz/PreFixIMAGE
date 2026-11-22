from __future__ import annotations
import base64
import hashlib
import json
import io
from pathlib import Path
from PIL import Image, ImageOps, UnidentifiedImageError
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from .models import NAMES
from .online import get_active_checkpoint
from .inference import run, status, MODE
from .metrics import image_metrics

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 10 * 1024 * 1024
MAX_SIDE = 1024  # change only after GPU memory testing
app = FastAPI(title='iMAGE — Image Denoiser', version='0.1.0')
app.mount('/static', StaticFiles(directory=ROOT / 'frontend'), name='static')


@app.get('/')
def index():
    return FileResponse(ROOT / 'frontend/main.html')


@app.get('/api/status')
def model_status():
    return status()


def load_image(data: bytes) -> Image.Image:
    if not data or len(data) > MAX_BYTES:
        raise HTTPException(413, f'Image required, maximum {MAX_BYTES // 1024 // 1024} MB')
    try:
        with Image.open(io.BytesIO(data)) as im:
            if im.width * im.height > 25_000_000:
                raise HTTPException(413, 'Image pixel count too large')
            im = ImageOps.exif_transpose(im).convert('RGB')
            im.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
            return im.copy()
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise HTTPException(415, 'Unsupported or corrupted image')


def png_data_uri(im: Image.Image) -> str:
    buffer = io.BytesIO()
    im.save(buffer, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')


def best_from_benchmark():
    """Recommend only when ALL 3 current deployed checkpoints were tested
    on the same full 500-image test set under identical settings."""
    summaries = []
    try:
        for name in NAMES:
            file = ROOT / 'results' / f'{name}_summary.json'
            summary = json.loads(file.read_text(encoding='utf8'))
            checkpoint = get_active_checkpoint(name)
            if not checkpoint.is_file() or summary.get('model') != name:
                return None
            if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != summary.get('checkpoint_sha256'):
                return None
            if summary.get('limited_test') or summary.get('number_of_test_images') != 500:
                return None
            summaries.append(summary)
        keys = ('sigma', 'patch', 'manifest_sha256')
        if any(any(record[k] != summaries[0][k] for k in keys) for record in summaries[1:]):
            return None
        best = max(summaries, key=lambda r: r['psnr_db'])['model']
        return best, (f'Highest mean PSNR across the same 500-image test set '
                      f'(sigma={summaries[0]["sigma"]}; {summaries[0]["patch"]} px evaluation crops).')
    except (OSError, KeyError, ValueError, TypeError):
        return None


@app.post('/api/denoise')
async def denoise(image: UploadFile = File(...), reference: UploadFile | None = File(None)):
    img = load_image(await image.read(MAX_BYTES + 1))
    ref = load_image(await reference.read(MAX_BYTES + 1)) if reference and reference.filename else None
    if ref and ref.size != img.size:
        raise HTTPException(422, 'Clean reference dimensions must match noisy input after resizing')
    outputs = []
    for name in NAMES:
        try:
            result, info = run(name, img)
            entry = {'key': name, 'image': png_data_uri(result), **info}
            if ref:
                entry['metrics'] = image_metrics(ref, result)
            outputs.append(entry)
        except Exception as exc:
            outputs.append({'key': name, 'error': f'{type(exc).__name__}: {exc}'})
    successful = [o for o in outputs if 'error' not in o]
    best = None
    best_reason = 'A best-model recommendation requires a clean reference or validated benchmark results.'
    if MODE == 'real':
        if ref and successful:
            candidates = [o for o in successful if o['metrics']['psnr'] is not None]
            best = max(candidates, key=lambda o: o['metrics']['psnr'])['key'] if candidates else None
            best_reason = 'Highest PSNR against the uploaded clean reference.'
        elif len(successful) == len(NAMES):
            rec = best_from_benchmark()
            if rec:
                best, best_reason = rec
    return {'mode': MODE, 'input': png_data_uri(img), 'reference_supplied': ref is not None,
            'size': {'width': img.width, 'height': img.height},
            'models': outputs, 'best': best, 'best_reason': best_reason}


@app.post('/api/denoise/stream')
async def denoise_stream(image: UploadFile = File(...), reference: UploadFile | None = File(None)):
    """Newline-delimited JSON events; actual per-model completion, one GPU job at a time."""
    img = load_image(await image.read(MAX_BYTES + 1))
    ref = load_image(await reference.read(MAX_BYTES + 1)) if reference and reference.filename else None
    if ref and ref.size != img.size:
        raise HTTPException(422, 'Clean reference dimensions must match noisy input after resizing')
    input_uri, _, _ = png_payload(img)

    def events():
        def emit(value):
            return json.dumps(value, separators=(',', ':')) + '\n'

        yield emit({'event': 'start', 'input': input_uri,
                    'size': {'width': img.width, 'height': img.height},
                    'mode': MODE, 'models': list(NAMES), 'reference_supplied': ref is not None})
        for name in NAMES:
            yield emit({'event': 'model_started', 'key': name})
            try:
                result, info = run(name, img)
                uri, output_bytes, _ = png_payload(result)
                payload = {'event': 'model_finished', 'key': name,
                           'image': uri, 'file_size_bytes': output_bytes, **info}
                if ref is not None:
                    payload['metrics'] = image_metrics(ref, result)
                yield emit(payload)
            except Exception as exc:
                yield emit({'event': 'model_failed', 'key': name,
                            'error': f'{type(exc).__name__}: {exc}'})
        yield emit({'event': 'done'})

    return StreamingResponse(events(), media_type='application/x-ndjson',
                             headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


def png_payload(im: Image.Image):
    buffer = io.BytesIO()
    im.save(buffer, format='PNG')
    raw = buffer.getvalue()
    return 'data:image/png;base64,' + base64.b64encode(raw).decode('ascii'), len(raw), raw
