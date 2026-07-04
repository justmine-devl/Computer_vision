import cv2
import numpy as np
from dataclasses import dataclass
from src.restoration.base_filter import BaseRestorationFilter

@dataclass
class MorphGuidedDesnowConfig:
    color_space: str = "hsv"

    v_threshold: float = 0.82
    s_threshold: float = 0.25

    opening_kernel: int = 3
    closing_kernel: int = 3
    median_kernel: int = 3

    use_bilateral: bool = True
    bilateral_d: int = 7
    sigma_color: float = 50.0
    sigma_space: float = 50.0

    use_guided_filter: bool = True
    guided_radius: int = 8
    guided_eps: float = 1e-2

    alpha: float = 0.7
    gamma: float = 1.0
    clahe_clip: float = 0.0

class MorphGuidedDesnowFilter(BaseRestorationFilter):
    def __init__(self, config: MorphGuidedDesnowConfig):
        self.config = config

    def restore(self, image_bgr, debug=False):
        # Return desnowed image, and keep same shape and type
        original = image_bgr.astype(np.float32) / 255.0
        
        # 1. Preprocessing & Snow Candidate Mask
        if self.config.color_space.lower() == "hsv":
            hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
            # V in [0, 1], S in [0, 1]
            V = hsv[:, :, 2]
            S = hsv[:, :, 1]
            snow_mask = ((V > self.config.v_threshold) & (S < self.config.s_threshold)).astype(np.uint8)
        else:
            snow_mask = np.zeros(original.shape[:2], dtype=np.uint8)

        # 2. Morphological Mask Refinement
        mask_refined = snow_mask.copy()
        if self.config.opening_kernel > 0:
            k = self.config.opening_kernel
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
            mask_refined = cv2.morphologyEx(mask_refined, cv2.MORPH_OPEN, kernel)
        if self.config.closing_kernel > 0:
            k = self.config.closing_kernel
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
            mask_refined = cv2.morphologyEx(mask_refined, cv2.MORPH_CLOSE, kernel)

        # 3. Snow Suppression
        if self.config.median_kernel > 0:
            k = self.config.median_kernel
            filtered = cv2.medianBlur((original * 255).astype(np.uint8), k).astype(np.float32) / 255.0
        else:
            filtered = original.copy()

        # 4. Edge-Preserving Refinement
        if self.config.use_guided_filter:
            try:
                # guidedFilter from cv2.ximgproc
                from cv2.ximgproc import guidedFilter
                # Original as guide
                refined = guidedFilter(guide=original, src=filtered, radius=self.config.guided_radius, eps=self.config.guided_eps)
            except ImportError:
                # Fallback if ximgproc is not available
                self.config.use_guided_filter = False
                
        if not self.config.use_guided_filter and self.config.use_bilateral:
            filtered_8u = (filtered * 255).astype(np.uint8)
            refined_8u = cv2.bilateralFilter(filtered_8u, self.config.bilateral_d, self.config.sigma_color, self.config.sigma_space)
            refined = refined_8u.astype(np.float32) / 255.0
        elif not self.config.use_guided_filter and not self.config.use_bilateral:
            refined = filtered

        # 5. Soft Mask and Blending
        soft_mask = cv2.GaussianBlur(mask_refined.astype(np.float32), (5, 5), 0)
        soft_mask = np.clip(soft_mask, 0, 1)
        soft_mask_3d = np.expand_dims(soft_mask, axis=-1)
        
        output = soft_mask_3d * (self.config.alpha * refined + (1 - self.config.alpha) * original) + (1 - soft_mask_3d) * original

        # 6. Optional Postprocessing
        if self.config.gamma != 1.0:
            output = np.power(output, self.config.gamma)
            
        output_8u = np.clip(output * 255, 0, 255).astype(np.uint8)
        
        if self.config.clahe_clip > 0.0:
            lab = cv2.cvtColor(output_8u, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=self.config.clahe_clip, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            output_8u = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            
        if debug:
            return output_8u, {
                "snow_mask_raw": (snow_mask * 255).astype(np.uint8),
                "mask_refined": (mask_refined * 255).astype(np.uint8),
                "snow_suppressed": (filtered * 255).astype(np.uint8),
                "refined": (refined * 255).astype(np.uint8),
                "soft_mask": (soft_mask * 255).astype(np.uint8)
            }
            
        return output_8u
