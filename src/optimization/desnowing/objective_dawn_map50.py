import numpy as np
import os
import cv2
import uuid
import yaml
from src.optimization.desnowing.search_space import sample_desnow_config
from src.restoration.morph_guided_desnow import MorphGuidedDesnowFilter

class ObjectiveDawnMap50:
    def __init__(self, dataset, yolo_evaluator, base_results_dir):
        self.dataset = dataset
        self.yolo_evaluator = yolo_evaluator
        self.base_results_dir = base_results_dir

    def __call__(self, trial):
        config = sample_desnow_config(trial)
        desnow = MorphGuidedDesnowFilter(config)
        
        # generate a unique config hash
        config_hash = str(uuid.uuid4().hex)[:8]
        trial_dir = os.path.join(self.base_results_dir, f"desnow_optuna_config3", config_hash)
        images_dir = os.path.join(trial_dir, "images")
        labels_dir = os.path.join(trial_dir, "labels")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        
        # We need to build a yolo dataset yaml for this trial
        trial_txt = os.path.join(trial_dir, "val.txt")
        with open(trial_txt, "w") as f_val:
            for sample in self.dataset:
                restored = desnow.restore(sample["image"])
                
                img_path = sample["image_path"]
                basename = os.path.basename(img_path)
                out_path = os.path.join(images_dir, basename)
                
                cv2.imwrite(out_path, restored)
                f_val.write(f"{out_path}\n")
                
                # Copy label if exists, else empty
                lbl_name = os.path.splitext(basename)[0] + ".txt"
                lbl_path = sample.get("label_path", None)
                out_lbl = os.path.join(labels_dir, lbl_name)
                if lbl_path and os.path.exists(lbl_path):
                    import shutil
                    shutil.copy2(lbl_path, out_lbl)
                else:
                    open(out_lbl, 'w').close()
                    
        yaml_path = os.path.join(trial_dir, "dataset.yaml")
        with open(yaml_path, "w") as f_yaml:
            f_yaml.write(f"path: {trial_dir}\n")
            f_yaml.write(f"train: val.txt\n")
            f_yaml.write(f"val: val.txt\n")
            # DAWN snow classes
            f_yaml.write("names:\n")
            coco_names = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane", 5: "bus", 6: "train", 7: "truck"}
            for k, v in coco_names.items():
                f_yaml.write(f"  {k}: {v}\n")
                
        # Evaluate using YOLO
        self.yolo_evaluator.data_yaml = yaml_path
        results = self.yolo_evaluator.evaluate(yaml_path)
        
        if not results:
            return 0.0
            
        map50 = results.get("map50", 0.0)
        return float(map50)
