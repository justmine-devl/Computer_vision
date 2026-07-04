import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error
import lpips
import torch
_lpips_model = None

def _get_lpips_model():
    global _lpips_model
    if _lpips_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _lpips_model = lpips.LPIPS(net='alex').to(device)
    return _lpips_model

def compute_psnr(restored: np.ndarray, clear: np.ndarray) -> float:
    return peak_signal_noise_ratio(clear, restored, data_range=255)

def compute_ssim(restored: np.ndarray, clear: np.ndarray) -> float:
    return structural_similarity(clear, restored, data_range=255, channel_axis=2)

def compute_ms_ssim(restored: np.ndarray, clear: np.ndarray) -> float:
    try:
        from pytorch_msssim import ms_ssim
        device = "cuda" if torch.cuda.is_available() else "cpu"
        r_t = torch.from_numpy(restored).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
        c_t = torch.from_numpy(clear).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
        return ms_ssim(r_t, c_t, data_range=1.0, size_average=True).item()
    except ImportError:
        return structural_similarity(clear, restored, data_range=255, channel_axis=2)

def compute_lpips(restored: np.ndarray, clear: np.ndarray) -> float:
    model = _get_lpips_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    r_rgb = restored[:, :, ::-1].astype(np.float32)
    c_rgb = clear[:, :, ::-1].astype(np.float32)
    r_t = torch.from_numpy(r_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    c_t = torch.from_numpy(c_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    r_t = (r_t / 255.0) * 2 - 1
    c_t = (c_t / 255.0) * 2 - 1
    
    with torch.no_grad():
        score = model(r_t, c_t).item()
    return score

def compute_mae(restored: np.ndarray, clear: np.ndarray) -> float:
    return np.mean(np.abs(clear.astype(np.float64) - restored.astype(np.float64)))

def compute_mse(restored: np.ndarray, clear: np.ndarray) -> float:
    return mean_squared_error(clear, restored)
