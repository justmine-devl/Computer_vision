import cv2
import numpy as np
from src.metrics.no_reference import compute_brisque

def objective_desand_config2_brisque(trial, filter_class, dataset_pairs):
    from src.restoration.rbcp_desand import RBCPDesandConfig
    
    config = RBCPDesandConfig(
        patch_size=trial.suggest_categorical("patch_size", [7, 11, 15, 21, 31]),
        omega=trial.suggest_float("omega", 0.70, 0.98),
        t0=trial.suggest_float("t0", 0.05, 0.35),
        top_percent=trial.suggest_float("top_percent", 0.0005, 0.01, log=True),
        blue_reverse_strength=trial.suggest_float("blue_reverse_strength", 0.5, 1.5),
        blue_gain=trial.suggest_float("blue_gain", 0.8, 1.5),
        wb_strength=trial.suggest_float("wb_strength", 0.0, 1.0),
        use_guided_filter=True,
        guided_radius=trial.suggest_int("guided_radius", 10, 80),
        guided_eps=trial.suggest_float("guided_eps", 1e-4, 1e-1, log=True),
        gamma=trial.suggest_float("gamma", 0.8, 1.3),
        clahe_clip=trial.suggest_float("clahe_clip", 0.0, 3.0),
    )
    
    restorer = filter_class(config)
    brisque_scores = []
    
    for img_path, _ in dataset_pairs:
        img = cv2.imread(img_path)
        if img is None: continue
            
        restored = restorer.restore(img)
        score = compute_brisque(restored)
        if not np.isnan(score):
            brisque_scores.append(score)
            
    if not brisque_scores:
        return float('inf')
        
    return np.mean(brisque_scores)
