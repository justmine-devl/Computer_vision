from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(root: Path, max_images: int = 0, seed: int = 42, exclude_tokens: tuple[str, ...] = ()) -> list[Path]:
    images: list[Path] = []
    if root.exists():
        for path in root.rglob("*"):
            try:
                if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                    lowered = str(path).lower()
                    if not any(token.lower() in lowered for token in exclude_tokens):
                        images.append(path)
            except OSError:
                continue
    images = sorted(images)
    rng = random.Random(seed)
    rng.shuffle(images)
    return images[:max_images] if max_images > 0 else images


def load_rgb_canvas(path: Path, size: int) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    img.thumbnail((size, size), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
    return np.asarray(canvas).astype(np.float32) / 255.0


def luminance(img: np.ndarray) -> np.ndarray:
    return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]


def fft_log_magnitude(gray: np.ndarray) -> np.ndarray:
    fft = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.log1p(np.abs(fft))
    return (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)


def radial_profile(mag: np.ndarray, bins: int = 80) -> np.ndarray:
    h, w = mag.shape
    y, x = np.indices((h, w))
    cy, cx = h // 2, w // 2
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r = r / (r.max() + 1e-8)
    idx = np.minimum((r * bins).astype(np.int32), bins - 1)
    values = np.zeros(bins, dtype=np.float64)
    counts = np.zeros(bins, dtype=np.float64)
    np.add.at(values, idx, mag)
    np.add.at(counts, idx, 1)
    return values / np.maximum(counts, 1)


def band_energy(gray: np.ndarray) -> dict[str, float]:
    mag = np.abs(np.fft.fftshift(np.fft.fft2(gray))) ** 2
    h, w = mag.shape
    y, x = np.indices((h, w))
    cy, cx = h // 2, w // 2
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / (np.sqrt(cx**2 + cy**2) + 1e-8)
    total = mag.sum() + 1e-8
    return {
        "low_energy": float(mag[r < 0.10].sum() / total),
        "mid_energy": float(mag[(r >= 0.10) & (r < 0.35)].sum() / total),
        "high_energy": float(mag[r >= 0.35].sum() / total),
    }


def gradient_map(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx**2 + gy**2)
    return grad / (grad.max() + 1e-8)


def dark_channel(img: np.ndarray, ksize: int = 15) -> np.ndarray:
    min_rgb = img.min(axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    return cv2.erode(min_rgb, kernel)


def image_statistics(img: np.ndarray) -> dict[str, float]:
    gray = luminance(img)
    hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    grad = gradient_map(gray)
    stats = {
        "mean_luma": float(gray.mean()),
        "std_luma": float(gray.std()),
        "mean_saturation": float((hsv[..., 1] / 255.0).mean()),
        "std_saturation": float((hsv[..., 1] / 255.0).std()),
        "mean_gradient": float(grad.mean()),
        "p95_gradient": float(np.percentile(grad, 95)),
        "mean_dark_channel": float(dark_channel(img).mean()),
    }
    stats.update(band_energy(gray))
    return stats

