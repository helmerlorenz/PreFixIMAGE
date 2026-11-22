from __future__ import annotations
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def image_metrics(reference: Image.Image, output: Image.Image) -> dict:
    if reference.size != output.size:
        raise ValueError('Reference image must have the same dimensions as the processed image')
    gt = np.asarray(reference.convert('RGB'), dtype=np.uint8)
    pred = np.asarray(output.convert('RGB'), dtype=np.uint8)
    psnr = float('inf') if np.array_equal(gt, pred) else float(peak_signal_noise_ratio(gt, pred, data_range=255))
    return {
        'psnr': round(psnr, 4) if np.isfinite(psnr) else None,
        'ssim': round(float(structural_similarity(gt, pred, data_range=255, channel_axis=-1)), 5)
    }
