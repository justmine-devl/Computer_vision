import optuna
import numpy as np
from src.restoration.bm3d_denoise import BM3DDenoiseFilter, BM3DDenoiseConfig
from src.metrics.full_reference import compute_ssim

def sample_bm3d_config(trial: optuna.Trial, noise_level):
    if noise_level == 15:
        sigma_low, sigma_high = 0.02, 0.10
    elif noise_level == 25:
        sigma_low, sigma_high = 0.05, 0.15
    elif noise_level == 50:
        sigma_low, sigma_high = 0.10, 0.25
    elif noise_level == "mixed":
        sigma_low, sigma_high = 0.02, 0.25
    else:
        raise ValueError(f"Unknown noise_level: {noise_level}")

    return BM3DDenoiseConfig(
        sigma_psd=trial.suggest_float("sigma_psd", sigma_low, sigma_high),
        stage_arg=trial.suggest_categorical("stage_arg", ["all"]),
        profile="default",
    )

def objective_config1_specialized_ssim(trial: optuna.Trial, noise_level, val_dataset):
    config = sample_bm3d_config(trial, noise_level)
    filt = BM3DDenoiseFilter(config)

    scores = []
    max_images = 3
    
    for i in range(min(len(val_dataset), max_images)):
        noisy, clean, image_id = val_dataset[i]
        denoised = filt.restore(noisy)
        scores.append(compute_ssim(denoised, clean))

    return float(np.mean(scores))
