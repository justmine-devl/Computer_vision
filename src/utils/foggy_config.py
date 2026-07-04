"""
foggy_config.py — Configuration for Foggy Cityscapes fine-tuning.

All arguments are compatible with the existing options.py structure so
that any existing utility that reads `opt` continues to work unchanged.

Default target: Kaggle Tesla T4 ×2 (16 GB × 2)
  patch_size=256  batch_size=8  grad_accum_steps=1  fp16=True
  → ~10 GB peak VRAM per GPU (one GPU used by default).
  To use both T4s see the note in the --num_gpus help string.

For RTX 4050 6 GB (laptop) override with:
  --batch_size 1 --patch_size 96 --grad_accum_steps 8 --lr 5e-5

Usage
-----
    from foggy_config import get_foggy_opts
    opt = get_foggy_opts()      # parsed from sys.argv
    # — or —
    opt = get_foggy_opts([])    # defaults only, useful for notebooks
"""

import argparse
from typing import Optional, List


def get_foggy_opts(args: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Return a Namespace with all fine-tuning hyper-parameters.

    Parameters
    ----------
    args : list of str, optional
        If None (default) reads sys.argv. Pass ``[]`` to get all defaults.
    """
    parser = argparse.ArgumentParser(
        description='HOGformer Foggy Cityscapes fine-tuning'
    )

    # ── CUDA / GPU ──────────────────────────────────────────────────────────
    parser.add_argument(
        '--cuda', type=int, default=0,
        help='CUDA device index (0 = first T4 on Kaggle).'
    )
    parser.add_argument(
        '--num_gpus', type=int, default=1,
        help=(
            'Number of GPUs to use. Default 1 (single T4). '
            'To use both Kaggle T4s, switch to the repo\'s train.py with '
            '--num_node 1 --num_gpus 2 and a Lightning DDP trainer, or run '
            'two separate experiments on each GPU with --cuda 0 / --cuda 1.'
        )
    )

    # ── Training schedule ───────────────────────────────────────────────────
    parser.add_argument(
        '--epochs', type=int, default=30,
        help='Total fine-tuning epochs. 30 is sufficient for ~1k images with a frozen encoder.'
    )
    parser.add_argument(
        '--batch_size', type=int, default=8,
        help=(
            'Per-GPU batch size. '
            '8 at patch_size=256 uses ~10 GB VRAM on T4 (fp16). '
            'For RTX 4050 6 GB use 1 with patch_size=96.'
        )
    )
    parser.add_argument(
        '--grad_accum_steps', type=int, default=1,
        help=(
            'Gradient accumulation steps. Effective batch = batch_size × grad_accum_steps. '
            'Set to 1 for T4 (batch_size=8 already sufficient). '
            'For RTX 4050 use 8 to reach effective batch 8 with batch_size=1.'
        )
    )
    parser.add_argument(
        '--patch_size', type=int, default=256,
        help=(
            'Spatial crop size for training patches. '
            '256 is the standard for image restoration and fits in T4 16 GB (fp16). '
            'For RTX 4050 6 GB use 96.'
        )
    )
    parser.add_argument(
        '--num_workers', type=int, default=4,
        help='DataLoader worker processes. 4 is optimal for Kaggle (2-core CPU per notebook).'
    )

    # ── Optimizer / scheduler ───────────────────────────────────────────────
    parser.add_argument(
        '--lr', type=float, default=1e-4,
        help=(
            'Peak learning rate. Scaled from 5e-5 (batch=1) to 1e-4 (batch=8) '
            'via the linear scaling rule (lr ∝ effective_batch_size). '
            'For RTX 4050 with batch=1 and accum=8 (eff. batch=8) use 5e-5–1e-4.'
        )
    )
    parser.add_argument(
        '--warmup_epochs', type=int, default=3,
        help='Linear LR warm-up epochs. 3 epochs is sufficient for ~1k images.'
    )
    parser.add_argument(
        '--eta_min', type=float, default=1e-7,
        help='Minimum LR at end of cosine decay.'
    )
    parser.add_argument(
        '--weight_decay', type=float, default=1e-4,
        help='AdamW weight decay.'
    )

    # ── Paths ────────────────────────────────────────────────────────────────
    parser.add_argument(
        '--pretrained_ckpt', type=str,
        default='ckpt/adair5d.ckpt',
        help='Path to the pretrained HOGformer Lightning checkpoint.'
    )
    parser.add_argument(
        '--data_root', type=str,
        default='data/foggy_cityscapes',
        help=(
            'Root of the Foggy Cityscapes dataset. '
            'Must contain train/input, train/target, val/input, val/target.'
        )
    )
    parser.add_argument(
        '--beta_filter', type=str, default='0.02',
        help=(
            'Load only foggy images for this beta value. '
            'Options: 0.005, 0.01, 0.02. Use "all" or empty string for all.'
        )
    )
    parser.add_argument(
        '--save_dir', type=str, default='finetune_checkpoints/foggy',
        help='Directory to save best_psnr.pth, best_ssim.pth, latest.pth.'
    )
    parser.add_argument(
        '--log_dir', type=str, default='logs/foggy_finetune',
        help='TensorBoard log directory.'
    )
    parser.add_argument(
        '--output_path', type=str, default='output/foggy_finetune/',
        help='Where to save restored images during evaluation.'
    )

    # ── Freezing strategy ────────────────────────────────────────────────────
    parser.add_argument(
        '--freeze_encoder', action='store_true', default=False,
        help=(
            'Freeze the entire encoder (patch_embed + encoder_level1/2/3 + '
            'down-samplers + skip embeds). Reduces VRAM and protects features. '
            'Recommended for <500 images. Unset for 1 000+ images.'
        )
    )
    parser.add_argument(
        '--freeze_latent', action='store_true', default=False,
        help='Also freeze the latent (bottleneck) transformer blocks.'
    )

    # ── Mixed precision / fp16 ───────────────────────────────────────────────
    parser.add_argument(
        '--fp16', action='store_true', default=True,
        help='Use automatic mixed precision (fp16). Saves ~1.5 GB VRAM.'
    )
    parser.add_argument(
        '--no_fp16', dest='fp16', action='store_false',
        help='Disable mixed precision (full fp32).'
    )

    # ── Early stopping ───────────────────────────────────────────────────────
    parser.add_argument(
        '--early_stop_patience', type=int, default=8,
        help='Stop training if PSNR has not improved for this many epochs (0 = disabled).'
    )

    # ── Logging ─────────────────────────────────────────────────────────────
    parser.add_argument(
        '--log_every', type=int, default=10,
        help='Log training metrics every N batches.'
    )
    parser.add_argument(
        '--val_every', type=int, default=1,
        help='Run validation every N epochs.'
    )

    # ── Resume ───────────────────────────────────────────────────────────────
    parser.add_argument(
        '--resume', type=str, default='',
        help=(
            'Path to a fine-tune checkpoint to resume from '
            '(e.g. finetune_checkpoints/foggy/latest.pth). '
            'Leave empty to start from --pretrained_ckpt.'
        )
    )

    # ── Degradation type for de_dict ─────────────────────────────────────────
    # kept for compatibility with anything that reads opt.de_type
    parser.add_argument(
        '--de_type', nargs='+', default=['dehaze'],
        help='Degradation type tag (keep as dehaze for fog).'
    )

    # ── data_file_dir kept for compat ─────────────────────────────────────────
    parser.add_argument(
        '--data_file_dir', type=str, default='data_dir/',
        help='Legacy: path to txt index files (not used by FoggyCityscapesDataset).'
    )

    namespace = parser.parse_args(args)

    # Normalise beta_filter
    if namespace.beta_filter.lower() in ('all', '', 'none'):
        namespace.beta_filter = None

    return namespace
'''
python train_foggy.py `
    --pretrained_ckpt ckpt/adair5d.ckpt `
    --data_root       data/foggy_cityscapes `
    --epochs          30 `
    --batch_size      2 `
    --patch_size      96 `
    --grad_accum_steps 4 `
    --lr              5e-5 `
    --fp16 `
    --early_stop_patience 6
'''