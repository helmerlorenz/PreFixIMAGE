"""Shared iMAGE dataset loader: clean/noisy RGB pairs, or legacy synthetic-clean manifests.

CSV columns for real pairs: source,split,noisy_path,clean_path,group_id
All paths relative to the manifest's parent folder. group_id links all crops/ISO
captures from the same original scene (unique inside each source).
"""
from __future__ import annotations
import csv
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset


def seed_for(filename: str, epoch: int = 0, seed: int = 42) -> int:
    b = f'{seed}|{filename}|{epoch}'.encode('utf8')
    return int.from_bytes(hashlib.sha256(b).digest()[:8], 'big') % (2**32)


def add_noise(clean: np.ndarray, sigma=25, seed=42):
    rng = np.random.default_rng(seed)
    return np.clip(clean.astype(np.float32) + rng.normal(0, sigma/255., clean.shape).astype(np.float32), 0, 1)



def secure_path(base: Path, relative: str) -> Path:
    base = base.resolve()
    relative_path = Path(relative)

    if relative_path.is_absolute():
        raise ValueError(
            f"Absolute paths are not allowed: {relative}"
        )

    path = (base / relative_path).resolve()

    if not path.is_relative_to(base):
        raise ValueError(
            f"Path outside project directory: {relative}"
        )

    if not path.is_file():
        raise FileNotFoundError(
            f"Image not found: {path}"
        )

    return path



class DenoiseDataset(Dataset):
    def __init__(self, manifest: str | Path, split: str, patch=128, sigma=25, seed=42, epoch=0):
        self.manifest = Path(manifest).resolve()
        with self.manifest.open(newline='', encoding='utf8') as file:
            reader = csv.DictReader(file)
            self.paired = 'noisy_path' in (reader.fieldnames or []) and 'clean_path' in (reader.fieldnames or [])
            if not self.paired and 'path' not in (reader.fieldnames or []):
                raise ValueError('Manifest needs paired noisy_path/clean_path or legacy path')
            self.files = [r for r in reader if r['split'] == split]
        if not self.files:
            raise ValueError(f'No {split} images found')
        self.split, self.patch, self.sigma, self.seed, self.epoch = split, patch, sigma, seed, epoch

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        entry = self.files[i]
        base = self.manifest.parent.parent
        clean_path = secure_path(base, entry['clean_path'] if self.paired else entry['path'])
        with Image.open(clean_path) as im:
            clean_im = ImageOps.exif_transpose(im).convert('RGB')
        if self.paired:
            noisy_path = secure_path(base, entry['noisy_path'])
            with Image.open(noisy_path) as im:
                noisy_im = ImageOps.exif_transpose(im).convert('RGB')
            if noisy_im.size != clean_im.size:
                raise ValueError(f'Paired images must be aligned and same dimensions: {noisy_path}, {clean_path}')
        if min(clean_im.size) < self.patch:
            scale = self.patch/min(clean_im.size)
            size=(round(clean_im.width*scale),round(clean_im.height*scale))
            clean_im=clean_im.resize(size,Image.Resampling.BICUBIC)
            if self.paired:
                noisy_im=noisy_im.resize(size,Image.Resampling.BICUBIC)
        identifier = entry['noisy_path'] if self.paired else entry['path']
        if self.split == 'train':
            rng=np.random.default_rng(seed_for(identifier,self.epoch,self.seed))
            x=int(rng.integers(0,clean_im.width-self.patch+1))
            y=int(rng.integers(0,clean_im.height-self.patch+1))
        else:
            x=(clean_im.width-self.patch)//2
            y=(clean_im.height-self.patch)//2
        bounds=(x,y,x+self.patch,y+self.patch)
        clean=np.asarray(clean_im.crop(bounds),dtype=np.float32)/255.
        if self.paired:
            noisy=np.asarray(noisy_im.crop(bounds),dtype=np.float32)/255.
        else:
            noise_seed=seed_for(identifier,self.epoch if self.split=='train' else 0,self.seed+11)
            noisy=add_noise(clean,sigma=self.sigma,seed=noise_seed)
        return (torch.from_numpy(noisy.copy()).permute(2,0,1),
                torch.from_numpy(clean.copy()).permute(2,0,1))
