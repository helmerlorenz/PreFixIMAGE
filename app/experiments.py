"""Reusable experiment methods called from Jupyter notebooks.
Train/validation/test are always sourced from the same immutable manifest.
These functions never silently substitute a filter for a neural network.
"""
from __future__ import annotations
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm.auto import tqdm

from .data import DenoiseDataset
from .models import ROOT, NAMES, create_model, load_checkpoint
from .online import get_active_checkpoint
from .inference import tiled_forward


def setup_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def dataset_counts(manifest: Path = ROOT / 'data/splits.csv'):
    """Validate the shared SIDD + PolyU 5000-pair manifest."""
    with Path(manifest).open(newline='', encoding='utf8') as file:
        rows = list(csv.DictReader(file))
    required = {'split', 'noisy_path', 'clean_path'}
    if rows and 'dataset' in rows[0]:
        source_key = 'dataset'
    elif rows and 'source' in rows[0]:
        source_key = 'source'
    else:
        raise ValueError('Manifest must contain a dataset or source column')
    if not rows or not required.issubset(rows[0]):
        raise ValueError('Manifest requires split, noisy_path, and clean_path columns')
    expected = {('SIDD', 'train'): 2800, ('SIDD', 'val'): 350, ('SIDD', 'test'): 350,
                ('PolyU', 'train'): 1200, ('PolyU', 'val'): 150, ('PolyU', 'test'): 150}
    actual = {key: 0 for key in expected}
    seen_pairs = set()
    for row in rows:
        key = (row[source_key], row['split'])
        if key not in actual:
            raise ValueError(f'Unexpected source or split: {key}')
        actual[key] += 1
        pair = (row[source_key], row['noisy_path'], row['clean_path'])
        if pair in seen_pairs:
            raise ValueError(f'Duplicate image pair: {pair}')
        seen_pairs.add(pair)
    if actual != expected:
        raise ValueError(f'Incorrect SIDD/PolyU quotas: {actual}')
    # The manifest has already undergone scene verification in notebook 01.
    counts = {s: sum(actual[(d, s)] for d in ('SIDD', 'PolyU')) for s in ('train', 'val', 'test')}
    return counts


def get_network(name: str, device=None, checkpoint: Path | str | None = None):
    if name not in NAMES:
        raise ValueError(f'Unknown model {name}')
    device = torch.device(device) if device else setup_device()
    network = create_model(name)
    source = Path(checkpoint) if checkpoint else get_active_checkpoint(name)
    load_checkpoint(network, name, source)
    return network.to(device), source


def validation_psnr(network, dataset, device, limit: int = 500):
    """Fixed noise/center crops on the validation split. No use of test set."""
    network.eval()
    scores = []
    with torch.inference_mode():
        for i, (noisy, clean) in enumerate(DataLoader(dataset, batch_size=1, shuffle=False)):
            if limit and i >= limit:
                break
            noisy = noisy.to(device)
            clean = clean.to(device)
            pred = network(noisy).clamp(0, 1)
            mse = ((pred-clean)**2).mean().clamp(min=1e-12)
            scores.append(float(-10 * torch.log10(mse)))
    if not scores:
        raise ValueError('Validation was empty')
    return float(np.mean(scores))


def fine_tune(name: str, *, epochs=5, batch_size=1, patch=128, sigma=25, lr=1e-5,
              validation_limit=500, max_train_steps=0, seed=42, patience=3, checkpoint=None):
    """Fine-tune a published checkpoint; never train from random weights by accident."""
    dataset_counts()
    torch.manual_seed(seed)
    device = setup_device()
    training = DenoiseDataset(ROOT/'data/splits.csv','train',patch=patch,sigma=sigma,seed=seed)
    validating = DenoiseDataset(ROOT/'data/splits.csv','val',patch=patch,sigma=sigma,seed=seed)
    train_loader = DataLoader(training,batch_size=batch_size,shuffle=True,num_workers=0,pin_memory=device.type=='cuda')
    net, source = get_network(name,device,checkpoint)
    print(f'{name}: fine-tuning on {device} from {source.name}; {len(training)} train, {len(validating)} validation')
    optimizer = torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=1e-5)
    scaler = torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    dest = ROOT/'weights'/f'{name}_finetuned.pth'; dest.parent.mkdir(exist_ok=True,parents=True)
    history=[]; best=-float('inf'); stale=0; start=time.perf_counter()
    for epoch in range(epochs):
        training.epoch=epoch
        net.train(); losses=[]
        for step,(noisy,clean) in enumerate(tqdm(train_loader,desc=f'{name} epoch {epoch+1}/{epochs}')):
            noisy,clean=noisy.to(device),clean.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,enabled=device.type=='cuda'):
                prediction=net(noisy)
                loss=nn.functional.l1_loss(prediction,clean)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(),1.0)
            scaler.step(optimizer);scaler.update()
            losses.append(float(loss.item()))
            if max_train_steps and step+1>=max_train_steps:
                break
        val=validation_psnr(net,validating,device,validation_limit)
        elapsed=(time.perf_counter()-start)/60
        row={'epoch':epoch+1,'train_l1':float(np.mean(losses)),
             'validation_psnr_db':val,'elapsed_min':round(elapsed,3)}
        history.append(row);print(row)
        if val>best+1e-4:
            best=val;stale=0
            torch.save({'state_dict':net.state_dict(), 'meta':{'name':name,'epoch':epoch+1,'sigma':sigma,
                       'validation_psnr':val,'training_time_min':elapsed,'seed':seed,
                       'training_samples':4000,'validation_samples':validation_limit or 500,
                       'max_train_steps':max_train_steps,'source':source.name}},dest)
        else:
            stale+=1
            if stale>=patience:
                print('Early stopping due to stagnant validation PSNR');break
    path=ROOT/'results'/f'{name}_training.csv';path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=history[0].keys());writer.writeheader();writer.writerows(history)
    del net
    if device.type=='cuda':torch.cuda.empty_cache()
    print(f'Best: {best:.3f} dB; saved {dest}. Training log: {path}')
    return history


def evaluate_model(name: str, *, checkpoint=None, patch=256, sigma=25, seed=42,
                   limit=0, tile=256):
    """Untouched test set; limit=0 means evaluate all 500 images.
    Timing uses model forward and GPU synchronization, excludes checkpoint loading.
    """
    dataset_counts()
    device=setup_device()
    ds=DenoiseDataset(ROOT/'data/splits.csv','test',patch=patch,sigma=sigma,seed=seed)
    net,source=get_network(name,device,checkpoint)
    net.eval()
    first=ds[0][0][None].to(device)
    with torch.inference_mode():
        for _ in range(2):tiled_forward(net,first,name,tile=tile)
    if device.type=='cuda':torch.cuda.synchronize()
    rows=[]
    with torch.inference_mode():
        for i,(noisy,clean) in enumerate(tqdm(DataLoader(ds,batch_size=1,shuffle=False),desc=f'Testing {name}')):
            if limit and i>=limit:break
            noisy=noisy.to(device)
            if device.type=='cuda':torch.cuda.synchronize()
            start=time.perf_counter()
            pred=tiled_forward(net,noisy,name,tile=tile)
            if device.type=='cuda':torch.cuda.synchronize()
            elapsed=(time.perf_counter()-start)*1000
            ground=clean[0].permute(1,2,0).numpy()
            actual=pred[0].permute(1,2,0).cpu().numpy()
            rows.append({'index':i,'psnr':float(peak_signal_noise_ratio(ground,actual,data_range=1)),
                         'ssim':float(structural_similarity(ground,actual,data_range=1,channel_axis=2)),
                         'runtime_ms':elapsed})
    out=ROOT/'results';out.mkdir(parents=True,exist_ok=True)
    with (out/f'{name}_test.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=['index','psnr','ssim','runtime_ms']);writer.writeheader();writer.writerows(rows)
    deployed=ROOT/'weights'/f'{name}.pth'
    if source==deployed or source==get_active_checkpoint(name):
        cksrc=source
    else:cksrc=source
    summary={'model':name,'checkpoint':str(cksrc),'checkpoint_sha256':hashlib.sha256(cksrc.read_bytes()).hexdigest(),
             'manifest_sha256':hashlib.sha256((ROOT/'data/splits.csv').read_bytes()).hexdigest(),
             'split':'test','number_of_test_images':len(rows),'limited_test':bool(limit),
             'sigma':sigma,'patch':patch,'seed':seed,'tile':tile,'device':str(device),
             'psnr_db':round(float(np.mean([r['psnr'] for r in rows])),4),
             'ssim':round(float(np.mean([r['ssim'] for r in rows])),5),
             'mean_runtime_ms':round(float(np.mean([r['runtime_ms'] for r in rows])),2),
             'parameters_m':round(sum(p.numel() for p in net.parameters())/1e6,3)}
    (out/f'{name}_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    del net
    if device.type=='cuda':torch.cuda.empty_cache()
    print(json.dumps(summary,indent=2))
    return summary


def summaries_as_dataframe():
    import pandas as pd
    results=[]
    for name in NAMES:
        f=ROOT/'results'/f'{name}_summary.json'
        if f.exists():results.append(json.loads(f.read_text(encoding='utf8')))
    return pd.DataFrame(results)
