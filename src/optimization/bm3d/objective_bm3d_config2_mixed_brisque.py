import optuna
import numpy as np
from src.restoration.bm3d_denoise import BM3DDenoiseFilter
from src.metrics.no_reference import compute_brisque
from src.optimization.bm3d.objective_bm3d_config1_specialized_ssim import sample_bm3d_config

def objective_config2_mixed_brisque(trial: optuna.Trial, val_datasets_dict):
    config = sample_bm3d_config(trial, noise_level="mixed")
    filt = BM3DDenoiseFilter(config)

    per_noise_scores = []
    max_images = 3

    for noise_level in [15, 25, 50]:
        val_dataset = val_datasets_dict[noise_level]
        scores = []
        for i in range(min(len(val_dataset), max_images)):
            noisy, _, image_id = val_dataset[i]
            denoised = filt.restore(noisy)
            scores.append(compute_brisque(denoised))
        per_noise_scores.append(np.mean(scores))

    return float(np.mean(per_noise_scores))
