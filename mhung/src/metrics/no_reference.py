import numpy as np
import pyiqa
import torch
import cv2
_models = {}

def _get_model(metric_name: str):
    global _models
    if metric_name not in _models:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        _models[metric_name] = pyiqa.create_metric(metric_name, device=device)
    return _models[metric_name]

def _compute_iqa(image_bgr: np.ndarray, metric_name: str) -> float:
    try:
        model = _get_model(metric_name)
        # Convert BGR to RGB
        img_rgb = image_bgr[:, :, ::-1].copy()
        # Convert numpy array to torch tensor [B, C, H, W] in [0, 1]
        img_t = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        img_t = img_t.to(device)
        
        with torch.no_grad():
            score = model(img_t).item()
        return score
    except Exception as e:
        print(f"Failed to compute {metric_name}: {e}")
        return float('nan')

def compute_brisque(image: np.ndarray) -> float:
    # Lower is better
    return _compute_iqa(image, 'brisque')

def compute_niqe(image: np.ndarray) -> float:
    # Lower is better
    return _compute_iqa(image, 'niqe')

def compute_piqe(image: np.ndarray) -> float:
    # Lower is better
    return _compute_iqa(image, 'piqe')

def compute_entropy(image: np.ndarray) -> float:
    # Supporting metric
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist / hist.sum()
    entropy = -np.sum(hist * np.log2(hist + 1e-7))
    return float(entropy)
