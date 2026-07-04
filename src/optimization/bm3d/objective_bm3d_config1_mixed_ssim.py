import numpy as np
from src.restoration.bm3d_denoise import BM3DDenoiseFilter
from src.metrics.full_reference import compute_ssim
from src.optimization.bm3d.objective_bm3d_config1_specialized_ssim import sample_bm3d_config

def objective_config1_mixed_ssim(trial, val_datasets_dict):
    config = sample_bm3d_config(trial, noise_level="mixed")
    filt = BM3DDenoiseFilter(config)

    per_noise_scores = []
    max_images = 3

    for noise_level in [15, 25, 50]:
        val_dataset = val_datasets_dict[noise_level]
        scores = []
        for i in range(min(len(val_dataset), max_images)):
            noisy, clean, image_id = val_dataset[i]
            denoised = filt.restore(noisy)
            scores.append(compute_ssim(denoised, clean))
        per_noise_scores.append(np.mean(scores))

    return float(np.mean(per_noise_scores))
