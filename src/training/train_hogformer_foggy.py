"""
train_foggy.py — Fine-tune HOGformer (Setting III/IV) on Foggy Cityscapes.

Default target: Kaggle Tesla T4 (16 GB, fp16)
----------------------------------------------
* batch_size=8   patch_size=256  → ~10 GB peak VRAM
* grad_accum_steps=1             → effective batch of 8
* fp16 (torch.cuda.amp)          → cuts activation memory ~40 %
* Encoder frozen, latent + decoder trainable (~32 % of params)
* LinearWarmupCosineAnnealingLR  matched to repo scheduler
* Loss = L1 + Pearson + HOG (same as train.py), HOG dropped last 5 epochs

RTX 4050 6 GB laptop override
------------------------------
  --batch_size 1 --patch_size 96 --grad_accum_steps 8 --lr 5e-5

Checkpoints saved
-----------------
  <save_dir>/best_psnr.pth   — best validation PSNR so far
  <save_dir>/best_ssim.pth   — best validation SSIM so far
  <save_dir>/latest.pth      — most recent epoch (always overwritten)

Usage
-----
  # Kaggle T4 (defaults)
  python train_foggy.py \\
      --pretrained_ckpt ckpt/adair5d.ckpt \\
      --data_root data/foggy_cityscapes

  # RTX 4050 6 GB override
  python train_foggy.py \\
      --pretrained_ckpt ckpt/adair5d.ckpt \\
      --data_root data/foggy_cityscapes \\
      --batch_size 1 --patch_size 96 --grad_accum_steps 8 --lr 5e-5

  # Resume from latest checkpoint
  python train_foggy.py --resume finetune_checkpoints/foggy/latest.pth
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "dl_nets" / "HOGformer"))

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
import math
import logging
from copy import deepcopy
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# ── repo imports ──────────────────────────────────────────────────────────────
from net.model import HOGformer
from utils.schedulers import LinearWarmupCosineAnnealingLR
from utils.foggy_dataset import build_foggy_datasets
from utils.foggy_config import get_foggy_opts

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('train_foggy')


# ---------------------------------------------------------------------------
# Loss functions  (mirror of train.py — kept self-contained)
# ---------------------------------------------------------------------------

class HOGLayer(nn.Module):
    """Differentiable HOG feature extractor (identical to train.py)."""

    def __init__(self, nbins=9, cell_size=8, block_size=2,
                 signed_gradient=False, eps=1e-8):
        super().__init__()
        self.nbins = nbins
        self.cell_size = cell_size
        self.block_size = block_size
        self.signed_gradient = signed_gradient
        self.eps = eps

        if not signed_gradient:
            angles = torch.tensor([i * math.pi / nbins for i in range(nbins)])
            self.bin_width = math.pi / nbins
        else:
            angles = torch.tensor([i * 2 * math.pi / nbins for i in range(nbins)])
            self.bin_width = 2 * math.pi / nbins

        self.register_buffer('angles', angles.view(1, -1, 1, 1))
        self.register_buffer('dx_filter',
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().view(1, 1, 3, 3))
        self.register_buffer('dy_filter',
            torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float().view(1, 1, 3, 3))

    def forward(self, x):
        B, C, H, W = x.shape
        if C == 3:
            gray = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1)
        else:
            gray = x

        # Always match BOTH device and dtype of the registered Sobel buffers.
        # dx_filter is float32 on whatever device this module lives on.
        # x may be float16 (AMP) on CUDA, or float32 on CPU during the HOG
        # loss call.  Casting here handles every combination cleanly.
        gray = gray.to(device=self.dx_filter.device, dtype=self.dx_filter.dtype)

        dx = F.conv2d(gray, self.dx_filter, padding=1)
        dy = F.conv2d(gray, self.dy_filter, padding=1)
        magnitude   = torch.sqrt(dx ** 2 + dy ** 2 + self.eps)
        orientation = torch.atan2(dy, dx + self.eps)
        if not self.signed_gradient:
            orientation = orientation.abs()

        delta = (orientation - self.angles).abs()
        if self.signed_gradient:
            delta = torch.min(delta, 2 * math.pi - delta)
        else:
            delta = torch.min(delta, math.pi - delta)
        weights = torch.relu(1.0 - delta / self.bin_width)

        new_H = (H // self.cell_size) * self.cell_size
        new_W = (W // self.cell_size) * self.cell_size
        if H % self.cell_size != 0 or W % self.cell_size != 0:
            magnitude = magnitude[:, :, :new_H, :new_W]
            weights   = weights  [:, :, :new_H, :new_W]

        hist = F.avg_pool2d(weights * magnitude,
                            kernel_size=self.cell_size, stride=self.cell_size)

        if self.block_size > 1:
            _, C2, Hc, Wc = hist.shape
            if Hc >= self.block_size and Wc >= self.block_size:
                blocks      = F.unfold(hist, kernel_size=self.block_size, stride=1)
                blocks      = blocks.permute(0, 2, 1).reshape(-1, C2 * self.block_size ** 2)
                block_norm  = blocks.norm(p=2, dim=1, keepdim=True)
                blocks      = blocks / (block_norm + self.eps)
                n_blocks    = (Hc - self.block_size + 1) * (Wc - self.block_size + 1)
                out = blocks.reshape(B, n_blocks, -1).reshape(B, -1)
            else:
                out = hist.reshape(B, -1)
        else:
            out = hist.reshape(B, -1)
        return out


class HOGLoss(nn.Module):
    def __init__(self, nbins=9, cell_size=8, block_size=2,
                 signed_gradient=False, loss_type='l1', eps=1e-8):
        super().__init__()
        self.hog = HOGLayer(nbins, cell_size, block_size, signed_gradient, eps)
        self.loss_type = loss_type

    def forward(self, pred, target):
        hp, ht = self.hog(pred), self.hog(target)
        if self.loss_type == 'l1':
            return F.l1_loss(hp, ht)
        return F.mse_loss(hp, ht)


def pearson_loss(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Pearson correlation loss — mirrors train.py compute_correlation_loss."""
    b = x1.shape[0]
    x1, x2 = x1.reshape(b, -1), x2.reshape(b, -1)
    x1m = x1 - x1.mean(dim=-1, keepdim=True)
    x2m = x2 - x2.mean(dim=-1, keepdim=True)
    num = (x1m * x2m).sum(dim=-1, keepdim=True)
    den = x1m.pow(2).sum(dim=-1, keepdim=True).sqrt() * \
          x2m.pow(2).sum(dim=-1, keepdim=True).sqrt()
    corr = num / (den + 1e-6)
    pearson = (1. - corr) / 2.
    mask = ~pearson.isnan() & ~pearson.isinf()
    return pearson[mask].mean() if mask.any() else torch.tensor(0., device=x1.device)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_psnr_ssim(restored: torch.Tensor,
                      clean: torch.Tensor) -> tuple:
    """Returns (avg_psnr, avg_ssim) for a batch."""
    r = restored.detach().cpu().clamp(0, 1).numpy().transpose(0, 2, 3, 1)
    c = clean.detach().cpu().clamp(0, 1).numpy().transpose(0, 2, 3, 1)
    psnr_sum = ssim_sum = 0.
    for ri, ci in zip(r, c):
        psnr_sum += peak_signal_noise_ratio(ci, ri, data_range=1)
        ssim_sum += structural_similarity(ci, ri, data_range=1, channel_axis=-1)
    n = r.shape[0]
    return psnr_sum / n, ssim_sum / n


# ---------------------------------------------------------------------------
# Freezing utilities
# ---------------------------------------------------------------------------

def _set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for p in module.parameters():
        p.requires_grad_(requires_grad)


def apply_freezing_strategy(net: HOGformer, opt) -> None:
    """
    Freezing strategy for ~1 000 images + 6 GB VRAM.

    Default (no flags):
      • Encoder levels 1, 2, 3 + patch_embed + downsamples + skip embeds → FROZEN
      • Latent (bottleneck) → TRAINABLE  ← learns fog-to-clean mapping
      • Entire decoder + refinement + output → TRAINABLE

    --freeze_encoder:  same as default (encoder frozen)
    --freeze_latent:   also freeze the bottleneck (only decoder trains)

    Justification
    -------------
    The encoder extracts low-level features (edges, textures) already
    well-learnt from diverse pretraining.  Freezing it:
      (a) saves ~1.2 GB VRAM (no encoder gradients stored)
      (b) prevents catastrophic forgetting of general features
      (c) forces the decoder to adapt to fog removal
    The latent holds high-level semantic features; keeping it trainable
    helps it specialise for fog distribution. Freeze only when VRAM is very
    tight or the dataset is tiny (<200 images).
    """
    # Always freeze encoder by default for 6 GB GPU
    encoder_modules = [
        net.patch_embed,
        net.encoder_level1,
        net.down1_2,
        net.encoder_level2,
        net.down2_3,
        net.encoder_level3,
        net.down3_4,
        net.skip_patch_embed1,
        net.skip_patch_embed2,
        net.skip_patch_embed3,
        net.reduce_chan_level_1,
        net.reduce_chan_level_2,
        net.reduce_chan_level_3,
    ]
    for m in encoder_modules:
        _set_requires_grad(m, False)

    if opt.freeze_latent:
        _set_requires_grad(net.latent, False)
        log.info('  ✗ latent (bottleneck): FROZEN')
    else:
        _set_requires_grad(net.latent, True)
        log.info('  ✓ latent (bottleneck): TRAINABLE')

    # Decoder + output are always trainable
    decoder_modules = [
        net.up4_3, net.reduce_chan_level3, net.decoder_level3,
        net.up3_2, net.reduce_chan_level2, net.decoder_level2,
        net.up2_1, net.decoder_level1,
        net.refinement,
        net.output,
    ]
    for m in decoder_modules:
        _set_requires_grad(m, True)

    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in net.parameters())
    log.info(f'  Trainable params: {trainable:,} / {total:,}  '
             f'({100 * trainable / total:.1f} %)')


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: str,
    net: HOGformer,
    optimizer: optim.Optimizer,
    scheduler,
    scaler: Optional[GradScaler],
    epoch: int,
    metrics: Dict,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch':            epoch,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict':    scaler.state_dict() if scaler else None,
        'metrics':          metrics,
    }, path)


def load_checkpoint(
    path: str,
    net: HOGformer,
    optimizer: optim.Optimizer,
    scheduler,
    scaler: Optional[GradScaler],
    device: torch.device,
) -> Dict:
    ckpt = torch.load(path, map_location=device)
    net.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    if scaler and ckpt.get('scaler_state_dict'):
        scaler.load_state_dict(ckpt['scaler_state_dict'])
    log.info(f'Resumed from {path}  (epoch {ckpt["epoch"]})')
    return ckpt


def load_pretrained_hogformer(ckpt_path: str, device: torch.device) -> HOGformer:
    """
    Load a HOGformer from a Lightning checkpoint (adair5d.ckpt format).
    The Lightning checkpoint stores the model under 'state_dict' with a
    'net.' prefix on every key.
    """
    net = HOGformer()

    ckpt = torch.load(ckpt_path, map_location='cpu')

    # Lightning wraps the model: keys look like "net.patch_embed.proj.weight"
    state = ckpt.get('state_dict', ckpt)
    # Strip "net." prefix if present
    new_state = {}
    for k, v in state.items():
        if k.startswith('net.'):
            new_state[k[4:]] = v
        else:
            new_state[k] = v

    missing, unexpected = net.load_state_dict(new_state, strict=False)
    if missing:
        log.warning(f'Missing keys ({len(missing)}): {missing[:5]} ...')
    if unexpected:
        log.warning(f'Unexpected keys ({len(unexpected)}): {unexpected[:5]} ...')
    log.info(f'Loaded pretrained HOGformer from {ckpt_path}')
    return net.to(device)


# ---------------------------------------------------------------------------
# Validation pass
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    net: HOGformer,
    val_loader: DataLoader,
    device: torch.device,
    fp16: bool,
) -> Dict[str, float]:
    net.eval()
    psnr_sum = ssim_sum = n_samples = 0

    for (_, _), foggy, clean in val_loader:
        foggy, clean = foggy.to(device), clean.to(device)
        with autocast(enabled=fp16):
            restored = net(foggy)
        p, s = compute_psnr_ssim(restored, clean)
        bs = foggy.shape[0]
        psnr_sum  += p * bs
        ssim_sum  += s * bs
        n_samples += bs

    net.train()
    return {
        'psnr': psnr_sum / n_samples,
        'ssim': ssim_sum / n_samples,
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main() -> None:
    opt = get_foggy_opts()

    # ── Device ──────────────────────────────────────────────────────────────
    device = torch.device(f'cuda:{opt.cuda}' if torch.cuda.is_available() else 'cpu')
    log.info(f'Device: {device}')
    if device.type == 'cuda':
        log.info(f'  GPU: {torch.cuda.get_device_name(device)}')
        log.info(f'  VRAM: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB')

    # ── Datasets ─────────────────────────────────────────────────────────────
    log.info('Loading datasets …')
    beta = opt.beta_filter if opt.beta_filter else None
    train_ds, val_ds = build_foggy_datasets(
        data_root=opt.data_root,
        patch_size=opt.patch_size,
        beta_filter=beta,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=min(opt.num_workers, 2),
        pin_memory=True,
        drop_last=False,
    )
    log.info(f'Train batches: {len(train_loader)}  Val samples: {len(val_ds)}')

    # ── Model ────────────────────────────────────────────────────────────────
    log.info('Building model …')
    net = load_pretrained_hogformer(opt.pretrained_ckpt, device)

    log.info('Applying freezing strategy:')
    log.info('  ✗ encoder (patch_embed + encoder_level1/2/3 + downs + skips): FROZEN')
    apply_freezing_strategy(net, opt)

    # ── Losses ──────────────────────────────────────────────────────────────
    # Both loss modules must be on the same device as the model so that their
    # registered buffers (Sobel filters in HOGLoss) reside on CUDA, not CPU.
    l1_fn  = nn.L1Loss().to(device)
    hog_fn = HOGLoss().to(device)

    # ── Optimizer / scheduler ────────────────────────────────────────────────
    trainable_params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        trainable_params,
        lr=opt.lr,
        weight_decay=opt.weight_decay,
        betas=(0.9, 0.999),
    )
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer=optimizer,
        warmup_epochs=opt.warmup_epochs,
        max_epochs=opt.epochs,
        warmup_start_lr=opt.eta_min,
        eta_min=opt.eta_min,
    )

    # ── fp16 scaler ─────────────────────────────────────────────────────────
    scaler = GradScaler() if opt.fp16 and device.type == 'cuda' else None
    log.info(f'Mixed precision fp16: {opt.fp16 and scaler is not None}')

    # ── TensorBoard ──────────────────────────────────────────────────────────
    os.makedirs(opt.log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=opt.log_dir)
    log.info(f'TensorBoard logs → {opt.log_dir}')
    log.info(f'  Run:  tensorboard --logdir {opt.log_dir}')

    # ── Checkpointing dirs ───────────────────────────────────────────────────
    os.makedirs(opt.save_dir, exist_ok=True)
    best_psnr_path = os.path.join(opt.save_dir, 'best_psnr.pth')
    best_ssim_path = os.path.join(opt.save_dir, 'best_ssim.pth')
    latest_path    = os.path.join(opt.save_dir, 'latest.pth')

    # ── Resume ───────────────────────────────────────────────────────────────
    start_epoch   = 0
    best_psnr     = -1.
    best_ssim     = -1.
    no_improve    = 0          # for early stopping

    if opt.resume and os.path.isfile(opt.resume):
        ckpt_data = load_checkpoint(
            opt.resume, net, optimizer, scheduler, scaler, device
        )
        start_epoch = ckpt_data['epoch'] + 1
        m = ckpt_data.get('metrics', {})
        best_psnr = m.get('best_psnr', best_psnr)
        best_ssim = m.get('best_ssim', best_ssim)

    # ── Training loop ────────────────────────────────────────────────────────
    log.info(
        f'\n{"─" * 60}\n'
        f'  Fine-tuning HOGformer on Foggy Cityscapes\n'
        f'  Epochs: {start_epoch}→{opt.epochs}   '
        f'LR: {opt.lr}   batch×accum: {opt.batch_size}×{opt.grad_accum_steps}\n'
        f'{"─" * 60}'
    )

    net.train()
    global_step = start_epoch * len(train_loader)

    for epoch in range(start_epoch, opt.epochs):
        epoch_t0   = time.time()
        loss_accum = 0.
        l1_accum   = 0.
        pear_accum = 0.
        hog_accum  = 0.

        optimizer.zero_grad(set_to_none=True)

        for batch_idx, ((_, _), foggy, clean) in enumerate(train_loader):
            foggy, clean = foggy.to(device), clean.to(device)

            # Forward + losses under autocast
            with autocast(enabled=(scaler is not None)):
                restored = net(foggy)
                l_l1   = l1_fn(restored, clean)
                l_pear = pearson_loss(restored, clean)

            # HOG loss is always computed in fp32 (Sobel buffers are float32).
            # Disabled in the last 5 epochs (matches train.py behaviour).
            if epoch < opt.epochs - 5:
                l_hog = hog_fn(restored.float(), clean.float())
                loss  = l_l1 + l_pear + l_hog
            else:
                l_hog = torch.zeros(1, device=device)
                loss  = l_l1 + l_pear

            # Scale by accum steps so effective LR matches batch×accum
            loss_scaled = loss / opt.grad_accum_steps

            # Backward
            if scaler:
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            # Accumulate metrics
            loss_accum += loss.item()
            l1_accum   += l_l1.item()
            pear_accum += l_pear.item()
            hog_accum  += l_hog.item()

            # Optimizer step every N batches
            is_accum_step = ((batch_idx + 1) % opt.grad_accum_steps == 0
                             or (batch_idx + 1) == len(train_loader))
            if is_accum_step:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            # Logging
            if (batch_idx + 1) % opt.log_every == 0:
                writer.add_scalar('train/loss',         loss.item(),     global_step)
                writer.add_scalar('train/loss_l1',      l_l1.item(),     global_step)
                writer.add_scalar('train/loss_pearson',  l_pear.item(),   global_step)
                writer.add_scalar('train/loss_hog',     l_hog.item(),    global_step)
                writer.add_scalar('train/lr',
                    optimizer.param_groups[0]['lr'], global_step)

            global_step += 1

        # Step LR scheduler (epoch-based, same as train.py)
        scheduler.step(epoch)

        # ── Epoch summary ─────────────────────────────────────────────────
        n_batches = len(train_loader)
        elapsed   = time.time() - epoch_t0
        lr_now    = optimizer.param_groups[0]['lr']
        log.info(
            f'Epoch {epoch + 1:3d}/{opt.epochs} '
            f'| loss {loss_accum / n_batches:.4f} '
            f'| l1 {l1_accum / n_batches:.4f} '
            f'| pearson {pear_accum / n_batches:.4f} '
            f'| hog {hog_accum / n_batches:.4f} '
            f'| lr {lr_now:.2e} '
            f'| {elapsed:.0f}s'
        )

        # ── Validation ────────────────────────────────────────────────────
        if (epoch + 1) % opt.val_every == 0:
            metrics = validate(net, val_loader, device, fp16=(scaler is not None))
            psnr_now, ssim_now = metrics['psnr'], metrics['ssim']

            log.info(
                f'  Val  PSNR: {psnr_now:.2f} dB  '
                f'SSIM: {ssim_now:.4f}  '
                f'[best PSNR: {best_psnr:.2f}  best SSIM: {best_ssim:.4f}]'
            )

            writer.add_scalar('val/psnr', psnr_now, epoch + 1)
            writer.add_scalar('val/ssim', ssim_now, epoch + 1)

            # Best PSNR checkpoint
            if psnr_now > best_psnr:
                best_psnr  = psnr_now
                no_improve = 0
                save_checkpoint(
                    best_psnr_path, net, optimizer, scheduler, scaler,
                    epoch, {'best_psnr': best_psnr, 'best_ssim': best_ssim}
                )
                log.info(f'  ✓ Saved best_psnr.pth  (PSNR {best_psnr:.2f} dB)')
            else:
                no_improve += 1

            # Best SSIM checkpoint (independent)
            if ssim_now > best_ssim:
                best_ssim = ssim_now
                save_checkpoint(
                    best_ssim_path, net, optimizer, scheduler, scaler,
                    epoch, {'best_psnr': best_psnr, 'best_ssim': best_ssim}
                )
                log.info(f'  ✓ Saved best_ssim.pth  (SSIM {best_ssim:.4f})')

            # Always save latest
            save_checkpoint(
                latest_path, net, optimizer, scheduler, scaler,
                epoch, {'best_psnr': best_psnr, 'best_ssim': best_ssim}
            )

            # ── Early stopping ─────────────────────────────────────────────
            if (opt.early_stop_patience > 0
                    and no_improve >= opt.early_stop_patience):
                log.info(
                    f'Early stopping triggered: PSNR has not improved for '
                    f'{no_improve} epochs.'
                )
                break

    # ── Final summary ────────────────────────────────────────────────────────
    writer.close()
    log.info(
        f'\nTraining complete.\n'
        f'  Best PSNR : {best_psnr:.2f} dB  → {best_psnr_path}\n'
        f'  Best SSIM : {best_ssim:.4f}      → {best_ssim_path}\n'
        f'  Latest    :                        → {latest_path}\n'
        f'\nTo evaluate:  python evaluate_foggy.py '
        f'--ckpt {best_psnr_path} --data_root {opt.data_root}'
    )


if __name__ == '__main__':
    main()
'''
$env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:128"
python train_foggy.py `
    --pretrained_ckpt ckpt/adair5d.ckpt `
    --data_root       data/foggy_cityscapes `
    --epochs          20 --batch_size 1 --patch_size 96 `
    --grad_accum_steps 8 --lr 5e-5 --fp16

Kaggle Fine-Tuning:
!python /kaggle/input/datasets/randomhuster/hogformer-finetuned-v2/HOGformer-finetuned-v2/HOGformer-main/settingIII_IV/train_foggy.py \
    --pretrained_ckpt /kaggle/input/datasets/randomhuster/hogformer-finetuned-v2/HOGformer-finetuned-v2/HOGformer-main/settingIII_IV/ckpt/adair5d.ckpt \
    --input_dir  /kaggle/input/datasets/suzukimio/allweather-testing/AllWeather_Testing/rain/input \
    --target_dir /kaggle/input/datasets/suzukimio/allweather-testing/AllWeather_Testing/rain/gt \
    --save_dir   /kaggle/working/finetune_checkpoints \
    --log_dir    /kaggle/working/logs/foggy_finetune \
    --fp16 \
    --patch_size 128 \
    --freeze_latent \
    --lr 1e-5 \
    --epochs 100 \
    --early_stop_patience 15
'''