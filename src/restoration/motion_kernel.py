from pathlib import Path

import cv2
import numpy as np

def make_motion_kernel(length: int, angle: float, size=None) -> np.ndarray:
    if size is None:
        size = length if length % 2 == 1 else length + 1

    kernel = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    half = length // 2
    kernel[center, center-half:center+half+1] = 1.0

    # Rotate kernel
    M = cv2.getRotationMatrix2D((center, center), angle, 1.0)
    kernel = cv2.warpAffine(kernel, M, (size, size), flags=cv2.INTER_CUBIC)
    
    kernel = np.maximum(kernel, 0)
    kernel = kernel / (kernel.sum() + 1e-8)
    return kernel

if __name__ == "__main__":
    debug_dir = Path("outputs") / "motion_deblur" / "debug" / "kernels"
    debug_dir.mkdir(parents=True, exist_ok=True)
    lengths = [7, 15, 31, 45]
    angles = [0, 30, 60, 90, 120, 150]
    for l in lengths:
        for a in angles:
            k = make_motion_kernel(l, a)
            vis = (k / k.max() * 255).astype(np.uint8)
            cv2.imwrite(str(debug_dir / f"kernel_length{l}_angle{a}.png"), vis)
    print(f"Kernel debug images saved to {debug_dir}")
