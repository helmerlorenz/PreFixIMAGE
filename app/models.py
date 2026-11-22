"""Model factories. Fetch upstream architectures automatically on first real inference."""
from __future__ import annotations
from pathlib import Path
from runpy import run_path
import sys
import types
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
NAMES = ('restormer', 'dncnn', 'nafnet')


class DnCNN(nn.Module):
    """Color-blind 20-layer DnCNN compatible with KAIR's released color checkpoint.

    KAIR merges batch normalization into convolutions for inference, so its
    dncnn_color_blind.pth uses Conv/ReLU blocks with learned biases.
    """
    def __init__(self, depth=20, channels=3, features=64):
        super().__init__()
        blocks = [nn.Conv2d(channels, features, 3, padding=1, bias=True), nn.ReLU(True)]
        for _ in range(depth - 2):
            blocks += [nn.Conv2d(features, features, 3, padding=1, bias=True), nn.ReLU(True)]
        blocks += [nn.Conv2d(features, channels, 3, padding=1, bias=True)]
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return x - self.model(x)


def _nafnet_factory(path: Path):
    """Load upstream architecture without importing all of legacy BasicSR.

    NAFNet_arch.py requires only LayerNorm2d and Local_Base on import.
    The optional NAFNetLocal conversion is not used here; we load NAFNet.
    """
    from torch.nn import functional as F

    class LayerNorm2d(nn.Module):
        def __init__(self, channels, eps=1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(channels))
            self.bias = nn.Parameter(torch.zeros(channels))
            self.eps = eps

        def forward(self, x):
            mean = x.mean(1, keepdim=True)
            var = ((x - mean) ** 2).mean(1, keepdim=True)
            return ((x - mean) * torch.rsqrt(var + self.eps) *
                    self.weight[None, :, None, None] + self.bias[None, :, None, None])

    class Local_Base:
        def convert(self, *args, **kwargs):
            raise RuntimeError('NAFNetLocal is not supported; use NAFNet.')

    module_names = ('basicsr', 'basicsr.models', 'basicsr.models.archs',
                    'basicsr.models.archs.arch_util', 'basicsr.models.archs.local_arch')
    saved = {name: sys.modules.get(name) for name in module_names}
    try:
        for name in module_names:
            if name not in sys.modules:
                mod = types.ModuleType(name)
                if name in module_names[:3]:
                    mod.__path__ = []
                sys.modules[name] = mod
        sys.modules['basicsr.models.archs.arch_util'].LayerNorm2d = LayerNorm2d
        sys.modules['basicsr.models.archs.local_arch'].Local_Base = Local_Base
        factory = run_path(str(path))['NAFNet']
    finally:
        for name, val in saved.items():
            if val is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = val
    return factory


def create_model(name: str) -> nn.Module:
    if name == 'dncnn':
        return DnCNN()
    if name == 'restormer':
        from .online import architecture_path
        arch = architecture_path('restormer')
        cls = run_path(str(arch))['Restormer']
        return cls(inp_channels=3, out_channels=3, dim=48, num_blocks=[4, 6, 6, 8],
                   num_refinement_blocks=4, heads=[1, 2, 4, 8],
                   ffn_expansion_factor=2.66, bias=False, LayerNorm_type='BiasFree',
                   dual_pixel_task=False)
    if name == 'nafnet':
        from .online import architecture_path
        arch = architecture_path('nafnet')
        cls = _nafnet_factory(arch)
        # Matches official SIDD width32 checkpoint. That checkpoint was trained on real camera noise.
        return cls(img_channel=3, width=32, enc_blk_nums=[2, 2, 4, 8],
                   middle_blk_num=12, dec_blk_nums=[2, 2, 2, 2])
    raise ValueError(f'Unknown model: {name}')


def checkpoint_path(name: str) -> Path:
    return ROOT / 'weights' / f'{name}.pth'


def load_checkpoint(model: nn.Module, name: str, path: Path | None = None) -> None:
    if path is None:
        from .online import get_active_checkpoint
        path = get_active_checkpoint(name)
    if not path.is_file():
        raise FileNotFoundError(f'Missing {name} weights: {path}')
    # Only load checkpoints from trusted authors or your own training runs.
    obj = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(obj, dict):
        raise ValueError(f'{path.name} is not a state dictionary')
    state = next((obj[k] for k in ('state_dict', 'params', 'params_ema') if k in obj), obj)
    if not isinstance(state, dict):
        raise ValueError(f'Invalid checkpoint structure in {path}')
    state = {k.removeprefix('module.').removeprefix('net_g.'): v for k, v in state.items()}
    # Color DnCNN checkpoint from KAIR uses model.* names and 20 conv layers.
    model.load_state_dict(state, strict=True)
