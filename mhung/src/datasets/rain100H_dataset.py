import os
import cv2

class PairedRainDataset:
    def __init__(self, split_file, base_dir=""):
        self.samples = []
        with open(split_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) == 2:
                    h, c = parts
                    self.samples.append({
                        "rainy_path": os.path.join(base_dir, h) if not os.path.isabs(h) else h,
                        "clean_path": os.path.join(base_dir, c) if not os.path.isabs(c) else c
                    })
                    
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        s = self.samples[idx]
        rainy = cv2.imread(s["rainy_path"])
        clean = cv2.imread(s["clean_path"])
        return {
            "rainy": rainy,
            "clean": clean,
            "rainy_path": s["rainy_path"],
            "clean_path": s["clean_path"]
        }
