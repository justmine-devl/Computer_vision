import os
import cv2
import shutil
import hashlib
from ultralytics import YOLO
from src.metrics.detection import extract_detection_metrics

def objective_desand_config3_map50(trial, filter_class, dataset_yaml, val_pairs, yolo_model_path="yolo26n.pt"):
    from src.restoration.rbcp_desand import RBCPDesandConfig
    
    config_dict = {
        "patch_size": trial.suggest_categorical("patch_size", [7, 11, 15, 21, 31]),
        "omega": trial.suggest_float("omega", 0.70, 0.98),
        "t0": trial.suggest_float("t0", 0.05, 0.35),
        "top_percent": trial.suggest_float("top_percent", 0.0005, 0.01, log=True),
        "blue_reverse_strength": trial.suggest_float("blue_reverse_strength", 0.5, 1.5),
        "blue_gain": trial.suggest_float("blue_gain", 0.8, 1.5),
        "wb_strength": trial.suggest_float("wb_strength", 0.0, 1.0),
        "use_guided_filter": True,
        "guided_radius": trial.suggest_int("guided_radius", 10, 80),
        "guided_eps": trial.suggest_float("guided_eps", 1e-4, 1e-1, log=True),
        "gamma": trial.suggest_float("gamma", 0.8, 1.3),
        "clahe_clip": trial.suggest_float("clahe_clip", 0.0, 3.0),
    }
    
    config = RBCPDesandConfig(**config_dict)
    
    # Hash config for caching
    config_str = str(sorted(config_dict.items()))
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    cache_dir = f"data/dawn/cache/rbcp_{config_hash}"
    cache_img_dir = os.path.join(cache_dir, "images")
    cache_lbl_dir = os.path.join(cache_dir, "labels")
    cache_yaml_path = os.path.join(cache_dir, "dataset.yaml")
    
    if not os.path.exists(cache_dir):
        os.makedirs(cache_img_dir, exist_ok=True)
        os.makedirs(cache_lbl_dir, exist_ok=True)
        
        restorer = filter_class(config)
        
        # We need to create a temporary yolo dataset for validation
        for img_path, lbl_path in val_pairs:
            img = cv2.imread(img_path)
            if img is not None:
                restored = restorer.restore(img)
                basename = os.path.basename(img_path)
                out_img = os.path.join(cache_img_dir, basename)
                cv2.imwrite(out_img, restored)
                
                # Copy label
                lbl_basename = os.path.basename(lbl_path)
                out_lbl = os.path.join(cache_lbl_dir, lbl_basename)
                if os.path.exists(lbl_path):
                    shutil.copy(lbl_path, out_lbl)
                    
        # Generate yaml for yolo evaluation on cache dir
        # We can copy the original dataset yaml and modify 'val' path
        with open(dataset_yaml, 'r') as f:
            lines = f.readlines()
            
        with open(cache_yaml_path, 'w') as f:
            for line in lines:
                if line.startswith("path:"):
                    f.write(f"path: {os.path.abspath(cache_dir)}\n")
                elif line.startswith("val:"):
                    f.write("val: images\n") # point to our cache images dir
                elif line.startswith("train:"):
                    f.write("train: images\n") # Ultralytics requires train key
                elif line.startswith("test:"):
                    f.write("test: images\n")
                else:
                    f.write(line)
                    
    # Evaluate with YOLO26
    model = YOLO(yolo_model_path)
    # ultralytics uses standard kwargs for val
    results = model.val(data=cache_yaml_path, save_json=False, save=False, plots=False, verbose=False)
    metrics = extract_detection_metrics(results)
    
    # Cleanup cache to save disk space if preferred, but Optuna might revisit? 
    # Optuna won't revisit exact config due to float spaces, so we can clean up
    shutil.rmtree(cache_dir)
    
    return metrics.get("map50", 0.0)
