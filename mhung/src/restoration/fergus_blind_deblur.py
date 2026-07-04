import numpy as np
from .wiener_deblur import WienerMotionDeblurFilter
from .richardson_lucy_deblur import RichardsonLucyMotionDeblurFilter
import sys
import os

# Add src to path to import metrics
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.metrics.no_reference import compute_brisque

class FergusBlindMotionDeblurFilter:
    def __init__(self, config):
        self.kernel_length_cands = config.get('kernel_length_candidates', [7, 11, 15, 21, 31])
        self.kernel_angle_cands = config.get('kernel_angle_candidates', [0, 30, 60, 90, 120, 150])
        self.deconv_method = config.get('deconv_method', 'wiener')
        self.reg_lambda = config.get('regularization_lambda', 0.01)
        self.num_iter = config.get('num_iter', 20)
        self.clip_output = config.get('clip_output', True)
        self.per_image_search = config.get('per_image_search', False)
        
        # Fixed best config (will use default if per_image_search is False)
        self.best_length = config.get('kernel_length', 15)
        self.best_angle = config.get('kernel_angle', 0.0)

    def apply(self, image: np.ndarray) -> np.ndarray:
        if not self.per_image_search:
            return self._run_deconv(image, self.best_length, self.best_angle)
            
        best_score = float('inf')
        best_restored = None
        
        # Simplified blind search
        for l in self.kernel_length_cands:
            for a in self.kernel_angle_cands:
                restored = self._run_deconv(image, l, a)
                score = compute_brisque(restored)
                if np.isnan(score):
                    score = float('inf')
                
                if score < best_score:
                    best_score = score
                    best_restored = restored
                    self.best_length = l
                    self.best_angle = a
                    
        return best_restored if best_restored is not None else image

    def _run_deconv(self, image, length, angle):
        if self.deconv_method == 'wiener':
            cfg = {
                'kernel_length': length, 
                'kernel_angle': angle, 
                'regularization_lambda': self.reg_lambda, 
                'clip_output': self.clip_output
            }
            f = WienerMotionDeblurFilter(cfg)
        else:
            cfg = {
                'kernel_length': length, 
                'kernel_angle': angle, 
                'num_iter': self.num_iter, 
                'clip_output': self.clip_output
            }
            f = RichardsonLucyMotionDeblurFilter(cfg)
        return f.apply(image)
