import os
import cv2
import numpy as np

class DawnDataset:
    def __init__(self, split_file: str, base_dir: str):
        self.base_dir = base_dir
        self.samples = []
        
        with open(split_file, "r") as f:
            lines = f.read().splitlines()
            
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split(',')
            image_path = parts[0]
            label_path = parts[1] if len(parts) > 1 and parts[1] else None
            
            self.samples.append({
                "image_path": image_path,
                "label_path": label_path,
                "image_rel": os.path.basename(image_path),
                "label_rel": os.path.basename(label_path) if label_path else None
            })
                
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = sample["image_path"]
        
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        return {
            "image_path": image_path,
            "label_path": sample["label_path"],
            "image_rel": sample["image_rel"],
            "label_rel": sample["label_rel"],
            "image": img
        }
