import os
import cv2
from pathlib import Path

class RainDropDataset:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        
        input_dir = Path(self.base_dir) / "input"
        restored_dir = Path(self.base_dir) / "restored"
        
        self.samples = []
        if input_dir.exists() and restored_dir.exists():
            for in_path in sorted(input_dir.glob("*.png")):
                res_path = restored_dir / in_path.name
                if res_path.exists():
                    self.samples.append({
                        "rainy": str(in_path),
                        "clean": str(res_path),
                        "rainy_rel": in_path.name
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        rainy_img = cv2.imread(sample["rainy"])
        clean_img = cv2.imread(sample["clean"])
        
        return {
            "rainy": rainy_img,
            "clean": clean_img,
            "image_path": sample["rainy"],
            "image_rel": sample["rainy_rel"]
        }

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
