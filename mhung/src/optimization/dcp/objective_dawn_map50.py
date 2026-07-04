import os
import shutil
import cv2
import hashlib
import json
from dataclasses import asdict

from src.optimization.dcp.search_space import sample_dcp_config
from src.restoration.dcp import DCPDehazeFilter

def hash_config(config):
    d = asdict(config)
    s = json.dumps(d, sort_keys=True)
    return hashlib.md5(s.encode('utf-8')).hexdigest()

class ObjectiveDawnMap50:
    def __init__(self, dataset, yolo_evaluator, base_results_dir):
        self.dataset = dataset
        self.yolo_evaluator = yolo_evaluator
        self.base_results_dir = base_results_dir

    def __call__(self, trial):
        config = sample_dcp_config(trial)
        config_hash = hash_config(config)

        restored_dir = os.path.join(self.base_results_dir, "restored", "optuna_config3", config_hash)
        restored_images_dir = os.path.join(restored_dir, "images")
        
        if not os.path.exists(restored_images_dir):
            os.makedirs(restored_images_dir, exist_ok=True)
            dcp = DCPDehazeFilter(config)
            
            for sample in self.dataset:
                restored = dcp.restore(sample["image"])
                basename = sample["image_rel"]
                out_path = os.path.join(restored_images_dir, basename)
                cv2.imwrite(out_path, restored)
                
                if sample.get("label_path") and os.path.exists(sample["label_path"]):
                    labels_dir = os.path.join(restored_dir, "labels")
                    os.makedirs(labels_dir, exist_ok=True)
                    label_out_path = os.path.join(labels_dir, sample["label_rel"])
                    shutil.copy2(sample["label_path"], label_out_path)
                
        temp_yaml_path = os.path.join(restored_dir, "temp_eval.yaml")
        lines = [
            f"path: A:/HUST_on_GitHub/ProjectCV/data/dawn",
            f"train: A:/HUST_on_GitHub/ProjectCV/data/dawn/images",
            f"val: {restored_images_dir}",
            f"names:",
            f"  0: person",
            f"  1: bicycle",
            f"  2: car",
            f"  3: motorcycle",
            f"  4: airplane",
            f"  5: bus",
            f"  6: train",
            f"  7: truck"
        ]
        with open(temp_yaml_path, "w") as f:
            f.write("\n".join(lines))

        # Evaluate
        self.yolo_evaluator.data_yaml = temp_yaml_path
        metrics = self.yolo_evaluator.evaluate(temp_yaml_path)

        return metrics["map50"]
