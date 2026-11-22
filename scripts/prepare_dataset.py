"""Create one exact, reproducible 4k/500/500 split. Run from project root."""
import argparse
import csv
import random
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def prepare(input_dir: Path, output: Path, count: int = 5000, seed: int = 42):
    input_dir = input_dir.resolve()
    data_dir = (ROOT / 'data').resolve()
    if not input_dir.is_relative_to(data_dir):
        raise ValueError(f'For reproducible paths, put images under {data_dir}')
    candidates = sorted(f for f in input_dir.rglob('*') if f.is_file()
                        and f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp', '.bmp'})
    valid = []
    for f in candidates:
        try:
            with Image.open(f) as image:
                if image.width >= 128 and image.height >= 128:
                    image.verify()
                    valid.append(f)
        except Exception:
            print('Skipping unreadable image:', f)
    if len(valid) < count:
        raise ValueError(f'Found {len(valid)} usable images; need at least {count}.')
    rng = random.Random(seed)
    rng.shuffle(valid)
    selected = valid[:count]
    n_train = int(count * 0.8)
    n_val = int(count * 0.1)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=['path', 'split'])
        writer.writeheader()
        for i, p in enumerate(selected):
            split = 'train' if i < n_train else 'val' if i < n_train + n_val else 'test'
            writer.writerow({'path': p.relative_to(output.parent.resolve()).as_posix(),
                             'split': split})
    print(f'Saved {output}: {n_train} train / {n_val} val / {count - n_train - n_val} test')
    print('All models must use this same manifest; do not regenerate it between model runs.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input-dir', type=Path, default=ROOT / 'data/raw/val2017')
    p.add_argument('--output', type=Path, default=ROOT / 'data/splits.csv')
    p.add_argument('--count', type=int, default=5000)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    prepare(args.input_dir, args.output, args.count, args.seed)
