import numpy as np
import cv2
from scipy.fftpack import dct, idct
from dataclasses import dataclass
from src.restoration.base_filter import BaseRestorationFilter

@dataclass
class BM3DDenoiseConfig:
    sigma_psd: float = 25 / 255
    image_range: str = "0_1"
    color_mode: str = "rgb"
    stage_arg: str = "all"
    profile: str = "default"

def dct3(group):
    # group shape: (N, block_size, block_size)
    d = dct(dct(group, axis=1, norm='ortho'), axis=2, norm='ortho')
    return dct(d, axis=0, norm='ortho')

def idct3(group):
    id = idct(idct(group, axis=0, norm='ortho'), axis=2, norm='ortho')
    return idct(id, axis=1, norm='ortho')

class BM3DDenoiseFilter(BaseRestorationFilter):
    def __init__(self, config: BM3DDenoiseConfig):
        self.config = config
        
    def restore(self, image_bgr):
        # We process in YCbCr to denoise luminance mostly, or channel-by-channel.
        # For simplicity and speed, we denoise each channel.
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img_float = image_rgb.astype(np.float32) / 255.0
        
        out_img = np.zeros_like(img_float)
        for ch in range(3):
            out_img[:, :, ch] = self._bm3d_channel(img_float[:, :, ch])
            
        out_img = np.clip(out_img, 0, 1)
        denoised_bgr = cv2.cvtColor((out_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        return denoised_bgr

    def _bm3d_channel(self, img_c):
        """
        Simplified BM3D Step 1 (Hard Thresholding) implemented from scratch.
        """
        sigma = self.config.sigma_psd
        block_size = 8
        search_window = 15  # Window size to search for similar blocks
        max_blocks = 16     # Max blocks in a 3D group
        step_size = 4       # Step size for reference blocks
        lambda_3d = 2.7
        threshold = lambda_3d * sigma
        
        h, w = img_c.shape
        out_img = np.zeros_like(img_c)
        weights = np.zeros_like(img_c)
        
        for i in range(0, h - block_size + 1, step_size):
            for j in range(0, w - block_size + 1, step_size):
                ref_block = img_c[i:i+block_size, j:j+block_size]
                
                # Search window boundaries
                rmin = max(0, i - (search_window - block_size)//2)
                rmax = min(h - block_size, i + (search_window - block_size)//2)
                cmin = max(0, j - (search_window - block_size)//2)
                cmax = min(w - block_size, j + (search_window - block_size)//2)
                
                distances = []
                coords = []
                # Step 2 for speed during search
                for r in range(rmin, rmax + 1, 2):
                    for c in range(cmin, cmax + 1, 2):
                        block = img_c[r:r+block_size, c:c+block_size]
                        dist = np.mean((ref_block - block)**2)
                        distances.append(dist)
                        coords.append((r, c))
                        
                sorted_idx = np.argsort(distances)[:max_blocks]
                selected_coords = [coords[idx] for idx in sorted_idx]
                
                # Build 3D group
                group = np.stack([img_c[r:r+block_size, c:c+block_size] for r, c in selected_coords])
                
                # 3D Transform
                group_dct = dct3(group)
                
                # Hard thresholding
                non_zero = np.abs(group_dct) > threshold
                group_dct = group_dct * non_zero
                
                # Weight
                nz_count = np.sum(non_zero)
                weight = 1.0 if nz_count == 0 else 1.0 / nz_count
                
                # Inverse 3D Transform
                group_idct = idct3(group_dct)
                
                # Aggregation
                for k, (r, c) in enumerate(selected_coords):
                    out_img[r:r+block_size, c:c+block_size] += group_idct[k] * weight
                    weights[r:r+block_size, c:c+block_size] += weight
                    
        # Average overlapping blocks
        out_img = np.divide(out_img, weights, out=out_img, where=weights>0)
        
        # Pixels that were not covered by blocks will remain original
        out_img[weights == 0] = img_c[weights == 0]
        
        return out_img
