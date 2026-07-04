import cv2
import numpy as np
from dataclasses import dataclass
from src.restoration.base_filter import BaseRestorationFilter

@dataclass
class LIMEDeLowlightConfig:
    illumination_floor: float = 0.05
    illumination_power: float = 0.75

    guided_radius: int = 16
    guided_eps: float = 1e-3

    exposure_gain: float = 1.0
    gamma: float = 1.0
    blend_alpha: float = 1.0

    use_clahe: bool = False
    clahe_clip: float = 1.5
    clahe_tile_grid_size: int = 8

    use_denoise: bool = False
    denoise_h: float = 3.0
    denoise_h_color: float = 3.0
    denoise_template_window: int = 7
    denoise_search_window: int = 21

    debug: bool = False

def box_filter(img, r):
    # radius r means kernel size 2*r + 1
    ksize = 2 * r + 1
    return cv2.boxFilter(img, -1, (ksize, ksize))

def guided_filter(guide, src, radius, eps):
    mean_I = box_filter(guide, radius)
    mean_p = box_filter(src, radius)
    mean_Ip = box_filter(guide * src, radius)
    
    cov_Ip = mean_Ip - mean_I * mean_p
    
    mean_II = box_filter(guide * guide, radius)
    var_I = mean_II - mean_I * mean_I
    
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    
    mean_a = box_filter(a, radius)
    mean_b = box_filter(b, radius)
    
    q = mean_a * guide + mean_b
    return q

class LIMEDeLowlightFilter(BaseRestorationFilter):
    def __init__(self, config: LIMEDeLowlightConfig):
        self.config = config

    def restore(self, image_bgr: np.ndarray) -> np.ndarray:
        # Stage 1 — Preprocessing
        image = image_bgr.astype(np.float32) / 255.0

        # Stage 2 — Initial Illumination Estimation
        T_hat = np.max(image, axis=2)

        if self.config.debug:
            self.debug_illumination_initial = (T_hat * 255).astype(np.uint8)

        # Stage 3 — Illumination Refinement
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) / 255.0
        T_refined = guided_filter(
            guide=gray,
            src=T_hat,
            radius=self.config.guided_radius,
            eps=self.config.guided_eps,
        )
        T_refined = np.clip(T_refined, self.config.illumination_floor, 1.0)

        if self.config.debug:
            self.debug_illumination_refined = (T_refined * 255).astype(np.uint8)

        # Stage 4 — Retinex-style Enhancement
        denom = np.power(T_refined[..., None], self.config.illumination_power)
        enhanced = image / np.maximum(denom, self.config.illumination_floor)
        enhanced = enhanced * self.config.exposure_gain
        enhanced = np.clip(enhanced, 0.0, 1.0)

        if self.config.debug:
            self.debug_enhanced_before_postprocess = (enhanced * 255).astype(np.uint8)

        # Stage 5 — Blend With Original Image
        output = self.config.blend_alpha * enhanced + (1 - self.config.blend_alpha) * image

        # Stage 6 — Optional Gamma Correction
        if self.config.gamma != 1.0:
            # Need to be careful with gamma: x^(1/gamma)
            # Add eps to avoid 0^gamma warning
            output = np.power(np.clip(output, 1e-6, 1.0), 1.0 / self.config.gamma)

        output_bgr = (np.clip(output, 0.0, 1.0) * 255.0).astype(np.uint8)

        # Stage 7 — Optional CLAHE
        if self.config.use_clahe:
            lab = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2LAB)
            L, A, B = cv2.split(lab)
            clahe = cv2.createCLAHE(
                clipLimit=self.config.clahe_clip, 
                tileGridSize=(self.config.clahe_tile_grid_size, self.config.clahe_tile_grid_size)
            )
            L2 = clahe.apply(L)
            output_bgr = cv2.cvtColor(cv2.merge([L2, A, B]), cv2.COLOR_LAB2BGR)

        # Stage 8 — Optional Denoise
        if self.config.use_denoise:
            output_bgr = cv2.fastNlMeansDenoisingColored(
                output_bgr,
                None,
                h=self.config.denoise_h,
                hColor=self.config.denoise_h_color,
                templateWindowSize=self.config.denoise_template_window,
                searchWindowSize=self.config.denoise_search_window,
            )

        if self.config.debug:
            self.debug_final_enhanced = output_bgr.copy()

        return output_bgr
