from __future__ import annotations
import os
import time
import threading
from PIL import Image, ImageFilter
import numpy as np
import torch
from torch.nn import functional as F
from .models import NAMES, create_model, load_checkpoint, checkpoint_path

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
MODE = os.environ.get('IMAGE_MODE', 'preview').lower()
if MODE not in ('preview', 'real'):
    raise ValueError('IMAGE_MODE must be preview or real')
LOCK = threading.Lock()  # prevent concurrent GPU inference/memory spikes in MVP
_CACHE = {}


def get_model(name):
    if name not in NAMES:
        raise ValueError('Unknown model')
    if name not in _CACHE:
        net = create_model(name)
        load_checkpoint(net, name)
        _CACHE[name] = net.to(DEVICE).eval()
    return _CACHE[name]


def status():
    return {'mode': MODE, 'device': str(DEVICE), 'models': [
        {'key': n, 'name': n.upper() if n == 'dncnn' else n.capitalize(),
         'checkpoint': 'deployed fine-tuned' if checkpoint_path(n).is_file() else 'online pretrained (auto-fetch)',
         'parameters_m': round(sum(p.numel() for p in _CACHE[n].parameters()) / 1e6, 3) if n in _CACHE else None}
        for n in NAMES]}


def _model_forward(net, x, name):
    """Run a model at original resolution (up to API size cap); pad only Restormer."""
    _, _, h, w = x.shape
    if name == 'restormer':
        dh, dw = (-h) % 8, (-w) % 8
        # replicate works also for small images
        x = F.pad(x, (0, dw, 0, dh), mode='replicate')
    with torch.inference_mode():
        result = net(x)
    return result[..., :h, :w].clamp(0, 1)


def tiled_forward(net, x, name, tile=256, overlap=32):
    """Overlapping tiles keep large web uploads from exhausting GPU memory.

    This is for UI inference only. Academic evaluations use fixed-sized crops
    through scripts/evaluate.py to keep speed and quality measurements consistent.
    """
    _, _, h, w = x.shape
    if h <= tile and w <= tile:
        return _model_forward(net, x, name)
    stride = tile - overlap
    if stride < 1:
        raise ValueError('Tile must be larger than overlap')

    def starts(length):
        if length <= tile:
            return [0]
        positions = list(range(0, length - tile + 1, stride))
        if positions[-1] != length - tile:
            positions.append(length - tile)
        return positions

    result = torch.zeros_like(x)
    counts = torch.zeros((1, 1, h, w), device=x.device, dtype=x.dtype)
    for y in starts(h):
        for xx in starts(w):
            patch = x[..., y:y + tile, xx:xx + tile]
            predicted = _model_forward(net, patch, name)
            ph, pw = predicted.shape[-2:]
            result[..., y:y + ph, xx:xx + pw] += predicted
            counts[..., y:y + ph, xx:xx + pw] += 1
    return result / counts.clamp_min(1)


def run(name: str, image: Image.Image):
    if name not in NAMES:
        raise ValueError('Unknown model')
    start = time.perf_counter()
    with LOCK:
        if MODE == 'preview':
            # Explicitly NOT the three ML models. Visual smoke testing only.
            filters = {'restormer': ImageFilter.GaussianBlur(1.5),
                       'dncnn': ImageFilter.MedianFilter(3),
                       'nafnet': ImageFilter.GaussianBlur(0.8)}
            result = image.filter(filters[name])
            return result, {'runtime_ms': round((time.perf_counter() - start) * 1000, 2),
                            'parameters_m': None, 'label': 'PREVIEW FILTER — NOT MODEL INFERENCE'}
        net = get_model(name)
        arr = np.asarray(image.convert('RGB'), dtype=np.float32) / 255.
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()
        begin = time.perf_counter()  # exclude first model load from inference latency
        out = tiled_forward(net, tensor, name, tile=int(os.getenv('IMAGE_TILE', '256')))
        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()
        elif DEVICE.type == 'mps':
            torch.mps.synchronize()
        elapsed = (time.perf_counter() - begin) * 1000
        outarr = (out[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
        return Image.fromarray(outarr), {
            'runtime_ms': round(elapsed, 2),
            'parameters_m': round(sum(p.numel() for p in net.parameters()) / 1e6, 3),
            'label': 'PRETRAINED / FINETUNED WEIGHTS'
        }
