import optuna
import numpy as np
from src.restoration.bm3d_denoise import BM3DDenoiseFilter
from src.metrics.no_reference import compute_brisque
from src.optimization.bm3d.objective_bm3d_config1_specialized_ssim import sample_bm3d_config

def objective_config2_specialized_brisque(trial: optuna.Trial, noise_level, val_dataset):
    config = sample_bm3d_config(trial, noise_level)
    filt = BM3DDenoiseFilter(config)

    scores = []
    max_images = 3
    
    for i in range(min(len(val_dataset), max_images)):
        noisy, _, image_id = val_dataset[i]
        
        denoised = filt.restore(noisy)
        scores.append(compute_brisque(denoised))

    return float(np.mean(scores))
