from dataclasses import dataclass
import numpy as np
import cv2

@dataclass
class DCPConfig:
    patch_size: int = 15
    omega: float = 0.95
    t0: float = 0.10
    top_percent: float = 0.001
    guided_radius: int = 40
    guided_eps: float = 1e-3
    gamma: float = 1.0
    clahe_clip: float = 0.0

class DCPDehazeFilter:
    def __init__(self, config: DCPConfig):
        self.config = config

    def _dark_channel(self, image):
        # image must be [0, 1] float
        min_channel = np.min(image, axis=2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.config.patch_size, self.config.patch_size))
        dark = cv2.erode(min_channel, kernel)
        return dark

    def _estimate_atmospheric_light(self, image, dark_channel):
        h, w = image.shape[:2]
        image_size = h * w
        num_pixels = int(max(image_size * self.config.top_percent, 1))

        dark_flat = dark_channel.reshape(-1)
        image_flat = image.reshape(-1, 3)

        indices = np.argpartition(dark_flat, -num_pixels)[-num_pixels:]
        
        # Calculate mean atmospheric light from top pixels
        A = np.mean(image_flat[indices], axis=0)
        return A

    def _estimate_transmission(self, image, A):
        norm_img = np.empty_like(image)
        for i in range(3):
            # Avoid division by zero
            A_c = A[i] if A[i] > 0 else 1e-6
            norm_img[:, :, i] = image[:, :, i] / A_c
            
        dark_norm = self._dark_channel(norm_img)
        transmission = 1 - self.config.omega * dark_norm
        return transmission

    def _guided_filter(self, I, p, r, eps):
        mean_I = cv2.boxFilter(I, cv2.CV_64F, (r, r))
        mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
        mean_Ip = cv2.boxFilter(I * p, cv2.CV_64F, (r, r))
        cov_Ip = mean_Ip - mean_I * mean_p

        mean_II = cv2.boxFilter(I * I, cv2.CV_64F, (r, r))
        var_I = mean_II - mean_I * mean_I

        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I

        mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
        mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))

        q = mean_a * I + mean_b
        return q

    def _refine_transmission(self, image, transmission):
        # Guided filter using gray scale image as guide
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        gray = gray.astype(np.float64) / 255.0
        
        refined = self._guided_filter(
            gray, 
            transmission, 
            self.config.guided_radius, 
            self.config.guided_eps
        )
        return refined

    def _recover(self, image, A, transmission):
        res = np.empty_like(image)
        t_bound = np.maximum(transmission, self.config.t0)
        
        for i in range(3):
            res[:, :, i] = (image[:, :, i] - A[i]) / t_bound + A[i]
            
        return np.clip(res, 0, 1)

    def _postprocess(self, image):
        res = image
        # Gamma correction
        if self.config.gamma != 1.0:
            res = np.power(res, self.config.gamma)
            
        # CLAHE
        if self.config.clahe_clip > 0:
            res_uint8 = (res * 255).astype(np.uint8)
            lab = cv2.cvtColor(res_uint8, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            clahe = cv2.createCLAHE(clipLimit=self.config.clahe_clip, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            lab = cv2.merge((l, a, b))
            res = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            res = res.astype(np.float64) / 255.0
            
        return res

    def restore(self, image_bgr: np.ndarray, return_debug=False) -> np.ndarray:
        """Restore one hazy BGR uint8 image and return BGR uint8 image."""
        if len(image_bgr.shape) != 3 or image_bgr.shape[2] != 3:
            raise ValueError("Input must be a 3-channel BGR image")
            
        # 1. Normalize
        img_float = image_bgr.astype(np.float64) / 255.0
        
        # 2. Dark channel
        dark = self._dark_channel(img_float)
        
        # 3. Atmospheric Light
        A = self._estimate_atmospheric_light(img_float, dark)
        
        # 4. Transmission map
        t = self._estimate_transmission(img_float, A)
        
        # 5. Refine transmission
        t_ref = self._refine_transmission(img_float, t)
        
        # 6. Recover
        recovered = self._recover(img_float, A, t_ref)
        
        # 7. Postprocess
        final = self._postprocess(recovered)
        
        # Convert back to uint8
        final_uint8 = (final * 255).astype(np.uint8)
        
        if return_debug:
            return final_uint8, {
                "dark_channel": (dark * 255).astype(np.uint8),
                "transmission_raw": (t * 255).astype(np.uint8),
                "transmission_refined": (t_ref * 255).astype(np.uint8),
            }
            
        return final_uint8
