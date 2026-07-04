from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, to_pil_image


def import_adair(adair_root: Path):
    root = str(adair_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    from net.model import AdaIR

    return AdaIR


def load_adair_checkpoint(model: torch.nn.Module, ckpt_path: Path) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt.get("params", ckpt))
    cleaned = {}
    for key, value in state.items():
        if key.startswith("net."):
            key = key[4:]
        cleaned[key] = value
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing or unexpected:
        print(f"AdaIR load_state_dict missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print("missing sample:", missing[:10])
        if unexpected:
            print("unexpected sample:", unexpected[:10])


def pad_to_multiple(x: torch.Tensor, multiple: int = 32) -> tuple[torch.Tensor, int, int]:
    h, w = x.shape[-2:]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return x, h, w


def restore_tensor(model: torch.nn.Module, x: torch.Tensor, tile: int = 0, overlap: int = 32) -> torch.Tensor:
    if tile <= 0:
        return model(x).clamp(0, 1)

    b, _, h, w = x.shape
    if b != 1:
        raise ValueError("Tiled inference supports batch size 1 only.")
    stride = max(tile - overlap, 1)
    out = torch.zeros_like(x)
    weight = torch.zeros_like(x)
    for top in range(0, h, stride):
        for left in range(0, w, stride):
            bottom = min(top + tile, h)
            right = min(left + tile, w)
            top0 = max(bottom - tile, 0)
            left0 = max(right - tile, 0)
            patch = x[:, :, top0:bottom, left0:right]
            patch, patch_h, patch_w = pad_to_multiple(patch, multiple=32)
            restored = model(patch).clamp(0, 1)[:, :, :patch_h, :patch_w]
            out[:, :, top0:bottom, left0:right] += restored
            weight[:, :, top0:bottom, left0:right] += 1.0
    return out / weight.clamp_min(1.0)


def restore_image(
    model: torch.nn.Module,
    image_path: Path,
    output_path: Path,
    device: torch.device,
    tile: int = 0,
    overlap: int = 32,
) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    x = pil_to_tensor(img).float().unsqueeze(0) / 255.0
    x, h, w = pad_to_multiple(x, multiple=32)
    x = x.to(device)
    with torch.no_grad():
        y = restore_tensor(model, x, tile=tile, overlap=overlap)
    y = y[:, :, :h, :w].cpu()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    to_pil_image(y.squeeze(0)).save(output_path)
    return y
