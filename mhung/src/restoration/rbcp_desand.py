import numpy as np
import cv2
from dataclasses import dataclass
from src.restoration.base_filter import BaseRestorationFilter

@dataclass
class RBCPDesandConfig:
    patch_size: int = 15
    omega: float = 0.95
    t0: float = 0.10
    top_percent: float = 0.001
    
    blue_reverse_strength: float = 1.0
    blue_gain: float = 1.10
    wb_strength: float = 0.5
    
    use_guided_filter: bool = True
    guided_radius: int = 40
    guided_eps: float = 1e-3
    
    gamma: float = 1.0
    clahe_clip: float = 0.0

class RBCPDesandFilter(BaseRestorationFilter):
    def __init__(self, config: RBCPDesandConfig):
        self.config = config
        
    def _gray_world_white_balance(self, img):
        """Gray-world white balance algorithm"""
        b, g, r = cv2.split(img)
        m_b, m_g, m_r = np.mean(b), np.mean(g), np.mean(r)
        m = (m_b + m_g + m_r) / 3.0
        
        # Avoid division by zero
        if m_b == 0 or m_g == 0 or m_r == 0:
            return img
            
        kb, kg, kr = m / m_b, m / m_g, m / m_r
        
        b = np.clip(b * kb, 0.0, 1.0)
        g = np.clip(g * kg, 0.0, 1.0)
        r = np.clip(r * kr, 0.0, 1.0)
        
        return cv2.merge([b, g, r])

    def restore(self, image_bgr: np.ndarray, debug_dir: str = None) -> np.ndarray:
        """
        Restore sand-dust image using Reversing Blue Channel Prior (RBCP).
        Args:
            image_bgr: Input BGR image (uint8, [H, W, 3])
            debug_dir: Optional directory to save intermediate outputs
        Returns:
            Restored BGR image (uint8, [H, W, 3])
        """
        H, W, _ = image_bgr.shape
        
        # 1. Preprocessing
        I = image_bgr.astype(np.float32) / 255.0
        
        if debug_dir is not None:
            import os
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(os.path.join(debug_dir, "input.png"), image_bgr)
            
        # 2. Reverse Blue Channel
        B = I[:, :, 0]
        G = I[:, :, 1]
        R = I[:, :, 2]
        
        B_rev = 1.0 - B
        B_rbcp = (1.0 - self.config.blue_reverse_strength) * B + self.config.blue_reverse_strength * B_rev
        I_rbcp = cv2.merge([B_rbcp, G, R])
        
        if debug_dir is not None:
            cv2.imwrite(os.path.join(debug_dir, "reversed_blue_channel.png"), (B_rbcp * 255).astype(np.uint8))
            
        # 3. RBCP Dark Channel
        min_channel = np.min(I_rbcp, axis=2)
        # Using minimum filter for patch size
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.config.patch_size, self.config.patch_size))
        D_rbcp = cv2.erode(min_channel, kernel)
        
        if debug_dir is not None:
            cv2.imwrite(os.path.join(debug_dir, "rbcp_dark_channel.png"), (D_rbcp * 255).astype(np.uint8))
            
        # 4. Atmospheric Light Estimation
        num_pixels = H * W
        num_top = max(1, int(num_pixels * self.config.top_percent))
        
        # Flatten and sort
        flat_D_rbcp = D_rbcp.flatten()
        indices = np.argsort(flat_D_rbcp)[-num_top:]
        
        # Use original image for atmospheric light estimation
        flat_I = I.reshape(-1, 3)
        top_I_pixels = flat_I[indices]
        
        # In the top pixels, choose the one with the highest intensity (L2 norm)
        intensities = np.linalg.norm(top_I_pixels, axis=1)
        max_idx = np.argmax(intensities)
        A = top_I_pixels[max_idx]
        
        # Avoid division by zero later
        A = np.maximum(A, 1e-6)
        
        if debug_dir is not None:
            import json
            with open(os.path.join(debug_dir, "atmospheric_light.json"), "w") as f:
                json.dump({"A_B": float(A[0]), "A_G": float(A[1]), "A_R": float(A[2])}, f)
                
        # 5. Coarse Transmission Estimation
        normalized_I_rbcp = I_rbcp / A
        min_channel_norm = np.min(normalized_I_rbcp, axis=2)
        D_norm = cv2.erode(min_channel_norm, kernel)
        
        t_coarse = 1.0 - self.config.omega * D_norm
        t_coarse = np.clip(t_coarse, 0.0, 1.0)
        
        if debug_dir is not None:
            cv2.imwrite(os.path.join(debug_dir, "transmission_coarse.png"), (t_coarse * 255).astype(np.uint8))
            
        # 6. Transmission Refinement
        if self.config.use_guided_filter:
            try:
                from cv2 import ximgproc
                gray_I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
                # cv2.ximgproc.createGuidedFilter takes guide image, radius, eps
                gf = ximgproc.createGuidedFilter(
                    guide=np.uint8(gray_I * 255), 
                    radius=self.config.guided_radius, 
                    eps=self.config.guided_eps * 255 * 255
                )
                t_refined = gf.filter(np.uint8(t_coarse * 255)) / 255.0
            except Exception as e:
                print(f"Warning: ximgproc guided filter failed: {e}. Falling back to original.")
                t_refined = t_coarse
        else:
            t_refined = t_coarse
            
        t_refined = np.clip(t_refined, 0.0, 1.0)
        
        if debug_dir is not None:
            cv2.imwrite(os.path.join(debug_dir, "transmission_refined.png"), (t_refined * 255).astype(np.uint8))
            
        # 7. Image Recovery
        t_safe = np.maximum(t_refined, self.config.t0)
        t_safe = np.expand_dims(t_safe, axis=2)
        
        J = (I - A) / t_safe + A
        J = np.clip(J, 0.0, 1.0)
        
        if debug_dir is not None:
            cv2.imwrite(os.path.join(debug_dir, "restored_before_color_correction.png"), (J * 255).astype(np.uint8))
            
        # 8. Color Correction
        # Blue channel compensation
        J[:, :, 0] = J[:, :, 0] * self.config.blue_gain
        J = np.clip(J, 0.0, 1.0)
        
        # Gray world white balance blending
        if self.config.wb_strength > 0:
            J_wb = self._gray_world_white_balance(J)
            J = (1.0 - self.config.wb_strength) * J + self.config.wb_strength * J_wb
            
        # Gamma correction
        if self.config.gamma != 1.0:
            J = np.power(J, self.config.gamma)
            
        # CLAHE
        if self.config.clahe_clip > 0:
            J_uint8 = (J * 255.0).astype(np.uint8)
            lab = cv2.cvtColor(J_uint8, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=self.config.clahe_clip, tileGridSize=(8,8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl,a,b))
            J = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR).astype(np.float32) / 255.0
            
        J = np.clip(J, 0.0, 1.0)
        
        if debug_dir is not None:
            cv2.imwrite(os.path.join(debug_dir, "restored_after_color_correction.png"), (J * 255).astype(np.uint8))
            
        # 9. Final Output
        output = np.clip(J * 255.0, 0, 255).astype(np.uint8)
        
        if debug_dir is not None:
            cv2.imwrite(os.path.join(debug_dir, "final_desand.png"), output)
            
        return output
