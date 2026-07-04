import os
import cv2
import numpy as np

class ResideDataset:
    def __init__(self, split_file: str, base_dir: str):
        self.base_dir = base_dir
        self.samples = []
        
        # Base dir is e.g. dataset/RESIDE-6K
        # The split file contains lines with "test/hazy/xxx.jpg test/GT/xxx.jpg" or similar, 
        # or just the basenames. Let's use relative paths in split_file like:
        # hazy_image_path,clear_image_path
        
        with open(split_file, "r") as f:
            lines = f.read().splitlines()
            
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) == 2:
                hazy_rel, clear_rel = parts
                self.samples.append({
                    "hazy_path": os.path.join(self.base_dir, hazy_rel),
                    "clear_path": os.path.join(self.base_dir, clear_rel)
                })
                
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        sample = self.samples[idx]
        hazy_path = sample["hazy_path"]
        clear_path = sample["clear_path"]
        
        hazy_img = cv2.imread(hazy_path)
        clear_img = cv2.imread(clear_path)
        
        if hazy_img is None:
            raise ValueError(f"Could not load hazy image: {hazy_path}")
        if clear_img is None:
            raise ValueError(f"Could not load clear image: {clear_path}")
            
        return {
            "hazy_path": hazy_path,
            "clear_path": clear_path,
            "hazy": hazy_img,
            "clear": clear_img
        }
