"""
evaluate_foggy.py — Compare pretrained vs fine-tuned HOGformer on Foggy Cityscapes.

Measures PSNR, SSIM, and LPIPS for:
  (A) Pretrained model  (adair5d.ckpt)
  (B) Fine-tuned model  (best_psnr.pth / best_ssim.pth / latest.pth)

Output
------
* Console table with per-image and aggregate metrics
* TXT report saved to <output_path>/eval_report.txt
* Restored images optionally saved to <output_path>/pretrained/ and <output_path>/finetuned/

Usage
-----
  # Compare both models on the val split
  python evaluate_foggy.py \\
      --pretrained_ckpt  ckpt/adair5d.ckpt \\
      --finetuned_ckpt   finetune_checkpoints/foggy/best_psnr.pth \\
      --data_root        data/foggy_cityscapes \\
      --split            val

  # Evaluate only the fine-tuned model (skip pretrained)
  python evaluate_foggy.py --finetuned_ckpt finetune_checkpoints/foggy/best_psnr.pth

  # Save restored images
  python evaluate_foggy.py --save_images --output_path output/eval/
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import logging
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from PIL import Image
from tqdm import tqdm

# ── repo imports ──────────────────────────────────────────────────────────────
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "dl_nets" / "HOGformer"))

from net.model import HOGformer
from utils.foggy_dataset import FoggyCityscapesDataset
from utils.image_utils import crop_img

# ── optional LPIPS ───────────────────────────────────────────────────────────
try:
    import lpips
    _LPIPS_AVAILABLE = True
except ImportError:
    _LPIPS_AVAILABLE = False
    warnings.warn(
        'lpips not installed — LPIPS score will be skipped.\n'
        'Install with:  pip install lpips',
        stacklevel=1,
    )

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('evaluate_foggy')


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def _load_hogformer_from_lightning(path: str, device: torch.device) -> HOGformer:
    """Load from Lightning checkpoint (adair5d.ckpt style)."""
    net = HOGformer()
    ckpt = torch.load(path, map_location='cpu')
    state = ckpt.get('state_dict', ckpt)
    new_state = {(k[4:] if k.startswith('net.') else k): v for k, v in state.items()}
    net.load_state_dict(new_state, strict=False)
    return net.to(device).eval()


def _load_hogformer_from_training(path: str, device: torch.device) -> HOGformer:
    """Load from train_foggy.py checkpoint (model_state_dict key)."""
    net = HOGformer()
    ckpt = torch.load(path, map_location='cpu')
    if 'model_state_dict' in ckpt:
        net.load_state_dict(ckpt['model_state_dict'], strict=True)
        log.info(f'Loaded fine-tuned checkpoint from epoch {ckpt.get("epoch", "?")}')
    else:
        # Fallback: try as raw state dict
        net.load_state_dict(ckpt, strict=False)
    return net.to(device).eval()


def load_model(path: str, device: torch.device) -> HOGformer:
    """Auto-detect checkpoint format and load accordingly."""
    ckpt = torch.load(path, map_location='cpu')
    if 'model_state_dict' in ckpt:
        return _load_hogformer_from_training(path, device)
    return _load_hogformer_from_lightning(path, device)


# ---------------------------------------------------------------------------
# LPIPS wrapper
# ---------------------------------------------------------------------------

class LPIPSMeter:
    def __init__(self, device: torch.device):
        self.enabled = _LPIPS_AVAILABLE
        if self.enabled:
            self.fn = lpips.LPIPS(net='alex').to(device)
            self.fn.eval()
        self.device = device

    @torch.no_grad()
    def compute(self, restored: torch.Tensor, clean: torch.Tensor) -> float:
        if not self.enabled:
            return float('nan')
        # LPIPS expects images in [-1, 1]
        r = restored.to(self.device).clamp(0, 1) * 2 - 1
        c = clean.to(self.device).clamp(0, 1) * 2 - 1
        return self.fn(r, c).mean().item()


# ---------------------------------------------------------------------------
# Evaluation loop for one model
# ---------------------------------------------------------------------------

class MetricsAccumulator:
    def __init__(self):
        self.psnr_vals:  List[float] = []
        self.ssim_vals:  List[float] = []
        self.lpips_vals: List[float] = []
        self.names:      List[str]   = []

    def update(self, name: str, psnr: float, ssim: float, lpips_val: float):
        self.names.append(name)
        self.psnr_vals.append(psnr)
        self.ssim_vals.append(ssim)
        self.lpips_vals.append(lpips_val)

    def summary(self) -> Dict[str, float]:
        return {
            'psnr_mean':  float(np.mean(self.psnr_vals)),
            'psnr_std':   float(np.std(self.psnr_vals)),
            'ssim_mean':  float(np.mean(self.ssim_vals)),
            'ssim_std':   float(np.std(self.ssim_vals)),
            'lpips_mean': float(np.nanmean(self.lpips_vals)),
            'lpips_std':  float(np.nanstd(self.lpips_vals)),
            'n':          len(self.psnr_vals),
        }


@torch.no_grad()
def evaluate_model(
    net: HOGformer,
    val_loader: DataLoader,
    device: torch.device,
    lpips_meter: LPIPSMeter,
    save_dir: Optional[str] = None,
    fp16: bool = True,
) -> MetricsAccumulator:
    """Run one full evaluation pass."""
    acc = MetricsAccumulator()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for (names, _), foggy, clean in tqdm(val_loader, leave=False, ncols=80):
        foggy, clean = foggy.to(device), clean.to(device)

        with autocast(enabled=(fp16 and device.type == 'cuda')):
            restored = net(foggy)

        restored_c = restored.clamp(0, 1)

        # Per-image metrics
        r_np = restored_c.cpu().numpy().transpose(0, 2, 3, 1)
        c_np = clean.cpu().numpy().transpose(0, 2, 3, 1)

        for i, name in enumerate(names):
            ri, ci = r_np[i], c_np[i]
            psnr = peak_signal_noise_ratio(ci, ri, data_range=1)
            ssim = structural_similarity(ci, ri, data_range=1, channel_axis=-1)
            lp   = lpips_meter.compute(restored_c[i:i+1], clean[i:i+1])
            acc.update(str(name), psnr, ssim, lp)

            if save_dir:
                img_arr = np.clip(ri * 255, 0, 255).astype(np.uint8)
                Image.fromarray(img_arr).save(
                    os.path.join(save_dir, f'{name}.png')
                )

    return acc


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def _fmt_row(label: str, summary: Dict) -> str:
    lpips_str = (
        f"{summary['lpips_mean']:.4f} ± {summary['lpips_std']:.4f}"
        if not np.isnan(summary['lpips_mean'])
        else 'N/A (install lpips)'
    )
    return (
        f"  {label:<20}  "
        f"PSNR: {summary['psnr_mean']:6.2f} ± {summary['psnr_std']:.2f} dB  |  "
        f"SSIM: {summary['ssim_mean']:.4f} ± {summary['ssim_std']:.4f}  |  "
        f"LPIPS: {lpips_str}  "
        f"(n={summary['n']})"
    )


def _delta_row(pre: Dict, ft: Dict) -> str:
    dp = ft['psnr_mean'] - pre['psnr_mean']
    ds = ft['ssim_mean'] - pre['ssim_mean']
    dl = ft['lpips_mean'] - pre['lpips_mean'] if not (
        np.isnan(ft['lpips_mean']) or np.isnan(pre['lpips_mean'])
    ) else float('nan')

    sign = lambda v: ('+' if v >= 0 else '')
    lpips_str = f"{sign(dl)}{dl:.4f}" if not np.isnan(dl) else 'N/A'
    return (
        f"  {'Δ (fine-tuned − pretrained)':<20}  "
        f"PSNR: {sign(dp)}{dp:6.2f} dB  |  "
        f"SSIM: {sign(ds)}{ds:.4f}  |  "
        f"LPIPS: {lpips_str}"
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Evaluate pretrained vs fine-tuned HOGformer on Foggy Cityscapes'
    )
    parser.add_argument('--pretrained_ckpt', type=str, default='ckpt/adair5d.ckpt')
    parser.add_argument('--finetuned_ckpt',  type=str,
                        default='finetune_checkpoints/foggy/best_psnr.pth')
    parser.add_argument('--data_root',  type=str, default='data/foggy_cityscapes')
    parser.add_argument('--split',      type=str, default='val',
                        choices=['train', 'val'])
    parser.add_argument('--patch_size', type=int, default=256,
                        help='Eval crop size (use full image by setting large value).')
    parser.add_argument('--beta_filter',type=str, default='0.02',
                        help='Beta filter for foggy images (0.005/0.01/0.02 or all).')
    parser.add_argument('--cuda',       type=int, default=0)
    parser.add_argument('--fp16',       action='store_true', default=True)
    parser.add_argument('--no_fp16',    dest='fp16', action='store_false')
    parser.add_argument('--save_images',action='store_true', default=False)
    parser.add_argument('--output_path',type=str, default='output/eval_foggy/')
    parser.add_argument('--num_workers',type=int, default=2)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    log.info(f'Evaluation device: {device}')

    dataset = FoggyCityscapesDataset(
        data_root=args.data_root,
        split=args.split,
        patch_size=args.patch_size,
        augment=False,
    )
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    log.info(f'Evaluating on {len(dataset)} images ({args.split} split)')

    lpips_meter = LPIPSMeter(device)

    results = {}

    # ── Pretrained ─────────────────────────────────────────────────────────
    if os.path.isfile(args.pretrained_ckpt):
        log.info(f'\n[1/2] Evaluating pretrained model: {args.pretrained_ckpt}')
        pre_net = load_model(args.pretrained_ckpt, device)
        pre_save = os.path.join(args.output_path, 'pretrained') if args.save_images else None
        results['pretrained'] = evaluate_model(
            pre_net, loader, device, lpips_meter,
            save_dir=pre_save, fp16=args.fp16
        )
        del pre_net
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    else:
        log.warning(f'Pretrained checkpoint not found: {args.pretrained_ckpt}')

    # ── Fine-tuned ──────────────────────────────────────────────────────────
    if os.path.isfile(args.finetuned_ckpt):
        log.info(f'\n[2/2] Evaluating fine-tuned model: {args.finetuned_ckpt}')
        ft_net = load_model(args.finetuned_ckpt, device)
        ft_save = os.path.join(args.output_path, 'finetuned') if args.save_images else None
        results['finetuned'] = evaluate_model(
            ft_net, loader, device, lpips_meter,
            save_dir=ft_save, fp16=args.fp16
        )
        del ft_net
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    else:
        log.warning(f'Fine-tuned checkpoint not found: {args.finetuned_ckpt}')

    # ── Report ───────────────────────────────────────────────────────────────
    sep = '─' * 100
    lines = [
        '',
        sep,
        f'  HOGformer Evaluation — Foggy Cityscapes  ({args.split} split)',
        sep,
    ]

    summaries = {}
    for key in ('pretrained', 'finetuned'):
        if key in results:
            s = results[key].summary()
            summaries[key] = s
            label = 'Pretrained' if key == 'pretrained' else 'Fine-tuned'
            lines.append(_fmt_row(label, s))

    if len(summaries) == 2:
        lines.append(_delta_row(summaries['pretrained'], summaries['finetuned']))

    lines += [sep, '']

    report = '\n'.join(lines)
    print(report)

    os.makedirs(args.output_path, exist_ok=True)
    report_path = os.path.join(args.output_path, 'eval_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    log.info(f'Report saved to {report_path}')

    if args.save_images:
        log.info(f'Restored images saved to {args.output_path}')


if __name__ == '__main__':
    main()
