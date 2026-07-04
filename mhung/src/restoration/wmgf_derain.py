import numpy as np
import cv2
from dataclasses import dataclass
from src.restoration.base_filter import BaseRestorationFilter

@dataclass
class WMGFConfig:
    noise_window_size: int = 3
    window_small: int = 3
    window_medium: int = 5
    window_large: int = 7
    noise_threshold_scale: float = 1.0
    spatial_sigma: float = 2.0
    range_sigma: float = 0.10
    guided_radius: int = 8
    guided_eps: float = 1e-2
    alpha: float = 1.0
    gamma: float = 1.0
    clahe_clip: float = 0.0

class WMGFDerainFilter(BaseRestorationFilter):
    def __init__(self, config: WMGFConfig):
        self.config = config

    def restore(self, image_bgr: np.ndarray, return_debug=False):
        # Stage 1 - Preprocessing
        img_float = image_bgr.astype(np.float32) / 255.0
        
        # Stage 2 - Rain/Noise Pixel Detection
        rain_mask = self._detect_rain(img_float)
        
        # Stage 3 & 4 - Adaptive Window Selection & Weighted Median Filtering
        coarse = self._adaptive_weighted_median(img_float, rain_mask)
        
        # Stage 5 - Guided Filtering
        # guide = coarse, src = original rainy image
        refined = self._guided_filter(coarse, img_float, self.config.guided_radius, self.config.guided_eps)
        
        # Stage 6 - Blending
        blended = self.config.alpha * refined + (1.0 - self.config.alpha) * img_float
        
        # Stage 7 - Postprocessing
        final = self._postprocess(blended)
        
        final_uint8 = np.clip(final * 255.0, 0, 255).astype(np.uint8)
        
        if return_debug:
            return final_uint8, {
                "rain_mask": (rain_mask * 255).astype(np.uint8),
                "coarse_wmf": np.clip(coarse * 255.0, 0, 255).astype(np.uint8),
                "guided_refined": np.clip(refined * 255.0, 0, 255).astype(np.uint8),
                "final_derained": final_uint8
            }
        return final_uint8

    def _detect_rain(self, img_float):
        # Convert to luminance
        gray = cv2.cvtColor(img_float, cv2.COLOR_BGR2GRAY)
        
        # Local mean and std
        w = self.config.noise_window_size
        mean = cv2.blur(gray, (w, w))
        sq_mean = cv2.blur(gray**2, (w, w))
        var = sq_mean - mean**2
        std = np.sqrt(np.maximum(var, 0))
        
        mask = np.abs(gray - mean) > self.config.noise_threshold_scale * std
        return mask.astype(np.uint8)

    def _adaptive_weighted_median(self, img_float, rain_mask):     
        img_uint8 = np.clip(img_float * 255.0, 0, 255).astype(np.uint8)
        
        # 1. Compute local noise density (ratio of rain pixels in medium window)
        mw = self.config.window_medium
        noise_density = cv2.boxFilter(rain_mask.astype(np.float32), cv2.CV_32F, (mw, mw))
        
        # 2. Compute median filters for all 3 window sizes
        med_small = cv2.medianBlur(img_uint8, self.config.window_small).astype(np.float32) / 255.0
        med_medium = cv2.medianBlur(img_uint8, self.config.window_medium).astype(np.float32) / 255.0
        med_large = cv2.medianBlur(img_uint8, self.config.window_large).astype(np.float32) / 255.0
        
        # 3. Create boolean masks for window selection based on noise density
        mask_small = (noise_density < 0.2).astype(np.float32)[..., np.newaxis]
        mask_medium = ((noise_density >= 0.2) & (noise_density < 0.5)).astype(np.float32)[..., np.newaxis]
        mask_large = (noise_density >= 0.5).astype(np.float32)[..., np.newaxis]
        
        # 4. Composite the adaptive median image
        adaptive_med = med_small * mask_small + med_medium * mask_medium + med_large * mask_large
        
        # 5. Apply the filtered result ONLY to the detected rain pixels
        rain_mask_3d = rain_mask[..., np.newaxis].astype(np.float32)
        out = img_float * (1.0 - rain_mask_3d) + adaptive_med * rain_mask_3d
        
        return out

    def _guided_filter(self, guide, src, radius, eps):
        try:
            return cv2.ximgproc.guidedFilter(guide=guide.astype(np.float32), src=src.astype(np.float32), radius=radius, eps=eps)
        except AttributeError:
            # Fallback Guided Filter for 3 channels
            res = np.empty_like(src)
            for i in range(3):
                I = guide[:, :, i]
                p = src[:, :, i]
                mean_I = cv2.boxFilter(I, cv2.CV_32F, (radius, radius))
                mean_p = cv2.boxFilter(p, cv2.CV_32F, (radius, radius))
                mean_Ip = cv2.boxFilter(I * p, cv2.CV_32F, (radius, radius))
                cov_Ip = mean_Ip - mean_I * mean_p
                
                mean_II = cv2.boxFilter(I * I, cv2.CV_32F, (radius, radius))
                var_I = mean_II - mean_I * mean_I
                
                a = cov_Ip / (var_I + eps)
                b = mean_p - a * mean_I
                
                mean_a = cv2.boxFilter(a, cv2.CV_32F, (radius, radius))
                mean_b = cv2.boxFilter(b, cv2.CV_32F, (radius, radius))
                
                res[:, :, i] = mean_a * I + mean_b
            return res

    def _postprocess(self, image):
        res = image
        # Gamma correction
        if self.config.gamma != 1.0:
            res = np.power(np.clip(res, 0, 1), self.config.gamma)
            
        # CLAHE
        if self.config.clahe_clip > 0:
            res_uint8 = np.clip(res * 255.0, 0, 255).astype(np.uint8)
            lab = cv2.cvtColor(res_uint8, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            clahe = cv2.createCLAHE(clipLimit=self.config.clahe_clip, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            lab = cv2.merge((l, a, b))
            res = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            res = res.astype(np.float32) / 255.0
            
        return res
