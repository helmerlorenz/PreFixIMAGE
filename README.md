# iMAGE — Image Denoiser | Notebook Edition

**One image denoising task, three pretrained modern models: Restormer, DnCNN, NAFNet.**

This version preserves the original iMAGE minimal dark theme. The website uses FastAPI for its backend; **all dataset preparation, fine-tuning, evaluation, visualization, and deployment are accessible through three VS Code Jupyter notebooks**.

## Important: online model imports and weights

**No manual Git cloning or `.pth` download is necessary.** On first use in **REAL** mode or via notebook 02, the application retrieves and caches:

| Model | Model architecture | Pretrained baseline weights |
|---|---|---|
| Restormer | Official [Restormer GitHub](https://github.com/swz30/Restormer/blob/main/basicsr/models/archs/restormer_arch.py) | [Official Restormer v1.0 `gaussian_color_denoising_blind.pth`](https://github.com/swz30/Restormer/releases/tag/v1.0) |
| DnCNN | KAIR-compatible DnCNN layers included in `app/models.py` | [Official KAIR release `dncnn_color_blind.pth`](https://github.com/cszn/KAIR/releases/tag/v1.0) |
| NAFNet | Official [NAFNet GitHub](https://github.com/megvii-research/NAFNet/blob/main/basicsr/models/archs/NAFNet_arch.py) | [Hugging Face mirror of official SIDD width32](https://huggingface.co/mikestealth/nafnet-models/blob/main/NAFNet-SIDD-width32.pth) (SHA-256 checked) |

Architectures are cached under `cache/upstream/`, weights under `cache/pretrained/`; both are ignored by Git. The first online load needs working internet and sufficient disk space. The combined weights exceed 200 MB. **Online importing still requires fetching model bytes locally; it eliminates the manual step, not the download itself.** A remote repository becoming unavailable, internet restrictions, or upstream changes can prevent first-time loading. Published model code is third-party code; only fetch from sources you trust.

**Honest accuracy note:** Restormer and DnCNN baselines are trained for Gaussian RGB noise; the NAFNet SIDD checkpoint was trained for real camera noise. This is not a fully matched pretrained baseline. Fine-tune on the shared Gaussian train split before claiming a controlled Gaussian-noise comparison.

## 1. VS Code setup (Windows PowerShell)

Install **Python 3.11**, VS Code **Python** extension and **Jupyter** extension. Extract the whole project ZIP. In VS Code: **File → Open Folder → `iMAGE_Notebook_Edition`**, then **Terminal → New Terminal**.

Run the following commands **one by one**:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

We use the `.venv` interpreter path explicitly so you **do not need to change PowerShell execution policy** or activate the environment. To use an NVIDIA GPU, install a **CUDA-enabled PyTorch** build based on the [official selector](https://pytorch.org/get-started/locally/). Verify with:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

The CPU-only version works for preview and limited smoke tests; fine-tuning Restormer and NAFNet realistically benefits from an NVIDIA GPU. No training duration or accuracy is guaranteed.

## 2. Open and test the website first

In your VS Code terminal (project root):

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

Open **http://127.0.0.1:8000**. This defaults to `preview` mode: three basic demonstration filters are clearly labeled and **are NOT Restormer, DnCNN, or NAFNet**. The website, image uploads, drag/drop/paste, results, download and optional clean-reference UI can be checked without a huge model download. Do not use Live Server or open `frontend/main.html` directly.

To use **real pretrained inference**, press `Ctrl+C` to stop the server, then:

```powershell
$env:IMAGE_MODE="real"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

The **first real upload** automatically fetches sources and pretrained weights (may take substantial time depending on connection). Subsequent runs use cached files. Errors are shown per-model; no filter masquerades as a model. The backend attempts all three models. Depending on available VRAM, large images and CPU Restormer inference may be slow. The upload API limits resolution to a maximum of 1,024 px on a side and tiles large images for memory safety.

To return to preview:

```powershell
$env:IMAGE_MODE="preview"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

## 3. Open Jupyter in VS Code

In the VS Code Explorer, open `notebooks/01_prepare_5000_images.ipynb`. Click **Select Kernel** (upper right) → **Python Environments** → choose `.venv` (Python 3.11). In the notebook, run one cell at a time with the triangular `▶` button. **Do not simply click Run All until you have read the `RUN_*` controls and dataset instructions.**

### Notebook 01 — Dataset preparation

A single **5,000 distinct clean-source-image dataset** is split **4,000 train / 500 val / 500 test**, using seed 42. Option A: provide 5,000+ image files inside `data/raw/val2017/`. Option B: the notebook can download and extract the public [COCO 2017 validation images](https://cocodataset.org/#download) (large archive, ~1 GB). In notebook 01 set `DOWNLOAD_COCO = True` for that option. No manual archive download is needed. The notebook writes `data/splits.csv` once; do not regenerate between models. An image and its synthetic noisy variant never cross splits.

The same deterministic Gaussian noise generation is used for all three models, starting with σ = 25/255. Training crops change consistently with epoch; validation and testing crops/noise are fixed. The dataset script checks there are exactly 5,000 readable source images at least 128 px wide and high.

### Notebook 02 — Fine-tuning (repeat for all three models)

Choose `MODEL='dncnn'`, then `'restormer'`, then `'nafnet'` in notebook 02. The notebook retrieves the corresponding **official model architecture** (where applicable) and **pretrained baseline weight files** automatically. Fine-tuning starts from those checkpoints; no training from random initialization. Start with a smoke test (`EPOCHS=1`, `MAX_TRAIN_STEPS=5`, `VALIDATION_LIMIT=10`), then reset to full workload (`EPOCHS=5`, `MAX_TRAIN_STEPS=0`, `VALIDATION_LIMIT=500`) and set `RUN_TRAINING=True`. The default full training pass uses all 4,000 train images per epoch. A single-model-at-a-time approach conserves GPU memory.

Checkpoints are saved under `weights/{model}_finetuned.pth`; CSV logs and training/validation graphs appear in `results/`. Choose epochs and learning rates based on **validation** data only; reserve the test data for final evaluation. Five epochs are a starting point, not proven optimum.

### Notebook 03 — Test and deployment

Set `USE_FINETUNED=True` after *all three* fine-tuned checkpoints exist; leave False to evaluate the published baseline models. Set `RUN_EVALUATION=True` and `LIMIT=0`. Notebook 03 evaluates all three models on the **identical 500-image test split** with PSNR, SSIM, inference time (CPU/GPU indicated), and parameter count. Produces `results/{model}_summary.json`, `results/{model}_test.csv`, and figures. Avoid re-testing during hyperparameter selection.

After full evaluation, if `USE_FINETUNED=True`, set `DEPLOY_FINETUNED=True` to copy checkpoints to `weights/{model}.pth`. Stop and restart the FastAPI server in REAL mode; it will use these instead of online baselines. A recommendation without a clean uploaded reference appears only when **all three current deployed checkpoints** have matching full 500-image evaluation summaries.

## 4. Files to understand first

```text
iMAGE_Notebook_Edition/
├─ START_HERE.txt                  ← Start here in VS Code
├─ frontend/
│  ├─ main.html                    ← Your dark UI
│  └─ app.js                       ← Browser interactions/API
├─ app/
│  ├─ main.py                      ← FastAPI routes
│  ├─ inference.py                 ← Real inference / honest preview
│  ├─ models.py                    ← Architecture adapters/checkpoint loader
│  ├─ online.py                    ← Online fetching and local caching
│  ├─ data.py                      ← Shared deterministic dataset loader
│  └─ experiments.py               ← Methods called by notebooks
├─ notebooks/
│  ├─ 01_prepare_5000_images.ipynb
│  ├─ 02_finetune_models.ipynb
│  └─ 03_evaluate_and_deploy.ipynb
├─ scripts/prepare_dataset.py     ← Helper imported by Notebook 01
├─ data/raw/                     ← Clean dataset images (not included)
├─ cache/                        ← Automatic upstream downloads (not included)
├─ weights/                      ← Fine-tuned/deployed weights (not included)
├─ results/                      ← Real experimental results (not included)
├─ requirements.txt
└─ tests/                        ← API smoke tests
```

## 5. Integrity and limitations

- **No fabricated metrics:** PSNR/SSIM for uploaded images require a matching *clean ground truth*. A normal noisy upload does not have a valid PSNR/SSIM score.
- **One fixed 5,000-image dataset:** 4,000/500/500, no leakage. The 500 test images should be held out throughout tuning.
- **Online sources** require internet on first real inference or training. Package size is small because trained models and dataset are NOT in the archive.
- **No published performance claims** should be made from a tiny smoke test. The three models are not yet benchmarked in this package.
- **Compatibility not fully verified:** The author's large checkpoints cannot be fetched or exercised in this sandbox (outbound networking is unavailable). Automated API, preview and notebook-structure tests passed, but actual remote model/weight compatibility must be verified on your machine. This is a starter implementation, not a claim that all real models have already loaded successfully.
- **Security:** pretrained checkpoint deserialization uses `torch.load(..., weights_only=True)` and the NAFNet mirror is SHA-256 checked; source files retrieved from the official GitHub repos are executed as Python, so review/trust them. For fully reproducible research, pin upstream Git commits/checksums after your first successful run.

## 6. Source references

- [Restormer code](https://github.com/swz30/Restormer) and [v1.0 pretrained release](https://github.com/swz30/Restormer/releases/tag/v1.0)
- [DnCNN / KAIR](https://github.com/cszn/KAIR)
- [NAFNet source](https://github.com/megvii-research/NAFNet), [SIDD checkpoint information](https://github.com/megvii-research/NAFNet/blob/main/docs/SIDD.md), and [HF mirror file](https://huggingface.co/mikestealth/nafnet-models/blob/main/NAFNet-SIDD-width32.pth)
- [COCO dataset](https://cocodataset.org/#download)
