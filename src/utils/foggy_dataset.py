"""
foggy_dataset.py — Foggy Cityscapes dataset loader for HOGformer Setting III/IV fine-tuning.

Folder layout expected on disk
-------------------------------
data/foggy_cityscapes/
    train/
        input/      ← foggy images   (0001.png … 1000.png)
        target/     ← clean GT images (0001.png … 1000.png)  ← SAME filenames
    val/
        input/
        target/

Naming convention
-----------------
Both input/ and target/ use identical numeric filenames:
    input/0001.png   ↔  target/0001.png
    input/0002.png   ↔  target/0002.png
    ...
    input/1000.png   ↔  target/1000.png

No renaming or regex stripping is required — pairing is done by matching
the exact filename in both directories.

The loader follows the same interface as DerainDehazeDataset:
  __getitem__ → ([name, de_type_id], degrad_tensor, clean_tensor)

Run this file directly to verify the dataset:
    python foggy_dataset.py --data_root data/foggy_cityscapes --split train
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import random
import argparse
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from utils.image_utils import crop_img


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}


def _is_image(fname: str) -> bool:
    return os.path.splitext(fname)[1].lower() in _EXTENSIONS


def _list_images(directory: str) -> List[str]:
    """Return sorted list of image filenames inside *directory*."""
    return sorted(
        f for f in os.listdir(directory) if _is_image(f)
    )


def _center_crop(img_np: np.ndarray, patch_size: int) -> np.ndarray:
    """Deterministic center crop — used for validation."""
    H, W = img_np.shape[:2]
    top  = max(0, (H - patch_size) // 2)
    left = max(0, (W - patch_size) // 2)
    return img_np[top:top + patch_size, left:left + patch_size]


def _random_crop_pair(
    img_a: np.ndarray,
    img_b: np.ndarray,
    patch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Crop both images at the same random location (H x W x C arrays)."""
    H, W = img_a.shape[:2]
    # If image is smaller than patch_size, fall back to center crop
    if H < patch_size or W < patch_size:
        return _center_crop(img_a, patch_size), _center_crop(img_b, patch_size)
    ind_H = random.randint(0, H - patch_size)
    ind_W = random.randint(0, W - patch_size)
    patch_a = img_a[ind_H:ind_H + patch_size, ind_W:ind_W + patch_size]
    patch_b = img_b[ind_H:ind_H + patch_size, ind_W:ind_W + patch_size]
    return patch_a, patch_b


def _random_augmentation(*arrays: np.ndarray) -> List[np.ndarray]:
    """
    Apply the same random flip / rotation to all arrays.
    Mirrors utils.image_utils.random_augmentation but inlined here so this
    file stays self-contained (the util function is identical in behaviour).
    """
    mode = random.randint(0, 7)
    result = []
    for arr in arrays:
        if   mode == 0: out = arr
        elif mode == 1: out = np.flipud(arr)
        elif mode == 2: out = np.rot90(arr)
        elif mode == 3: out = np.flipud(np.rot90(arr))
        elif mode == 4: out = np.rot90(arr, k=2)
        elif mode == 5: out = np.flipud(np.rot90(arr, k=2))
        elif mode == 6: out = np.rot90(arr, k=3)
        else:           out = np.flipud(np.rot90(arr, k=3))
        result.append(out.copy())
    return result


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FoggyCityscapesDataset(Dataset):
    """
    Foggy dataset loader for HOGformer fine-tuning.

    Expects images named ``0001.png`` … ``1000.png`` (or any consistent
    numeric / arbitrary names) where **input and target filenames are
    identical**.  No renaming or suffix-stripping is performed.

    Parameters
    ----------
    data_root : str
        Root path, e.g. ``"data/foggy_cityscapes"``.
        Must contain ``<split>/input/`` and ``<split>/target/`` sub-folders.
    split : str
        ``'train'`` or ``'val'``.
    patch_size : int
        Spatial size of each training patch (default 128).
        For validation the patch is a deterministic center crop.
    augment : bool
        Whether to apply random flip/rotation augmentation (train only).
    """

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        patch_size: int = 128,
        augment: bool = True,
    ) -> None:
        super().__init__()

        assert split in ('train', 'val'), f"split must be 'train' or 'val', got {split!r}"

        self.split      = split
        self.patch_size = patch_size
        self.augment    = augment and (split == 'train')
        self.to_tensor  = ToTensor()

        input_dir  = os.path.join(data_root, split, 'input')
        target_dir = os.path.join(data_root, split, 'target')

        if not os.path.isdir(input_dir):
            raise FileNotFoundError(
                f"Input directory not found: {input_dir}\n"
                "Make sure you have created the folder structure:\n"
                "  <data_root>/<split>/input/   ← foggy images\n"
                "  <data_root>/<split>/target/  ← clean GT images"
            )
        if not os.path.isdir(target_dir):
            raise FileNotFoundError(f"Target directory not found: {target_dir}")

        # Collect and pair images by identical filename.
        # input/0001.png  ↔  target/0001.png
        self.samples: List[Tuple[str, str]] = []   # (foggy_path, clean_path)
        input_names  = _list_images(input_dir)
        target_names = set(_list_images(target_dir))

        missing_gt = []
        for fname in input_names:
            if fname in target_names:
                self.samples.append(
                    (os.path.join(input_dir,  fname),
                     os.path.join(target_dir, fname))
                )
            else:
                missing_gt.append(fname)

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No paired images found.\n"
                f"  input dir : {input_dir}  ({len(input_names)} files)\n"
                f"  target dir: {target_dir}  ({len(target_names)} files)\n"
                f"  First few input names : {input_names[:5]}\n"
                f"  First few target names: {sorted(target_names)[:5]}\n"
                f"Make sure both directories contain files with IDENTICAL names "
                f"(e.g. 0001.png in input/ and 0001.png in target/)."
            )

        if missing_gt:
            print(f"[FoggyCityscapesDataset] Warning: {len(missing_gt)} input image(s) "
                  f"had no matching GT (e.g. '{missing_gt[0]}'). They are skipped.")

        print(
            f"[FoggyCityscapesDataset] split={split!r}  "
            f"pairs={len(self.samples)}  patch_size={patch_size}"
        )

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        foggy_path, clean_path = self.samples[idx]

        # Load images → uint8 HxWxC numpy arrays
        foggy_img = crop_img(
            np.array(Image.open(foggy_path).convert('RGB')), base=16
        )
        clean_img = crop_img(
            np.array(Image.open(clean_path).convert('RGB')), base=16
        )

        # Crop
        if self.split == 'train':
            foggy_patch, clean_patch = _random_crop_pair(
                foggy_img, clean_img, self.patch_size
            )
        else:
            # Deterministic center crop for reproducible validation metrics
            foggy_patch = _center_crop(foggy_img, self.patch_size)
            clean_patch = _center_crop(clean_img, self.patch_size)

        # Augmentation (train only)
        if self.augment:
            foggy_patch, clean_patch = _random_augmentation(foggy_patch, clean_patch)

        # → float tensors in [0, 1]
        foggy_tensor = self.to_tensor(foggy_patch)
        clean_tensor = self.to_tensor(clean_patch)

        # Return in the same format as DerainDehazeDataset:
        #   ([name, de_type_id], degraded_tensor, clean_tensor)
        # de_type_id = 4  (same as dehaze — fog is a haze variant)
        name = os.path.splitext(os.path.basename(foggy_path))[0]
        return [name, 4], foggy_tensor, clean_tensor


# ---------------------------------------------------------------------------
# Convenience builder used by train_foggy.py
# ---------------------------------------------------------------------------

def build_foggy_datasets(
    data_root: str,
    patch_size: int = 128,
    beta_filter: Optional[str] = None,   # kept for API compat; unused
) -> Tuple[FoggyCityscapesDataset, FoggyCityscapesDataset]:
    """Return (train_dataset, val_dataset)."""
    train_ds = FoggyCityscapesDataset(
        data_root=data_root,
        split='train',
        patch_size=patch_size,
        augment=True,
    )
    val_ds = FoggyCityscapesDataset(
        data_root=data_root,
        split='val',
        patch_size=patch_size,
        augment=False,
    )
    return train_ds, val_ds


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Verify FoggyCityscapesDataset')
    parser.add_argument('--data_root', type=str, default='data/foggy_cityscapes')
    parser.add_argument('--split',     type=str, default='train')
    parser.add_argument('--patch_size',type=int, default=128)
    args = parser.parse_args()

    ds = FoggyCityscapesDataset(
        data_root=args.data_root,
        split=args.split,
        patch_size=args.patch_size,
    )
    print(f"Total samples in '{args.split}' split: {len(ds)}")
    (name, de_id), foggy, clean = ds[0]
    print(f"Sample 0 : name={name!r}  de_id={de_id}")
    print(f"  foggy  : shape={foggy.shape}  range=[{foggy.min():.3f}, {foggy.max():.3f}]")
    print(f"  clean  : shape={clean.shape}  range=[{clean.min():.3f}, {clean.max():.3f}]")
    # Verify last sample too
    (name_last, _), _, _ = ds[-1]
    print(f"Sample -1 : name={name_last!r}")
    print("Dataset OK ✓")
