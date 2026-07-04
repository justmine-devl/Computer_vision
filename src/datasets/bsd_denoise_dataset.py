import os
import cv2

class BSDDenoiseDataset:
    def __init__(self, root: str, noise_level, split: str):
        self.root = root
        self.noise_level = noise_level
        self.split = split
        
        split_file = os.path.join(self.root, f"{self.split}.txt")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")
            
        with open(split_file, "r") as f:
            self.image_names = [line.strip() for line in f.read().splitlines() if line.strip()]
            
        self.samples = []
        
        if self.noise_level == "mixed":
            levels = [15, 25, 50]
            for lvl in levels:
                for img_name in self.image_names:
                    self.samples.append({
                        "name": img_name,
                        "noise_level": lvl,
                        "noisy_path": os.path.join(self.root, f"noise{lvl}", "noisy", img_name),
                        "clean_path": os.path.join(self.root, f"noise{lvl}", "clean", img_name)
                    })
        else:
            for img_name in self.image_names:
                self.samples.append({
                    "name": img_name,
                    "noise_level": int(self.noise_level),
                    "noisy_path": os.path.join(self.root, f"noise{self.noise_level}", "noisy", img_name),
                    "clean_path": os.path.join(self.root, f"noise{self.noise_level}", "clean", img_name)
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        noisy_bgr = cv2.imread(sample["noisy_path"])
        clean_bgr = cv2.imread(sample["clean_path"])
        
        if noisy_bgr is None:
            raise ValueError(f"Could not load noisy image: {sample['noisy_path']}")
        if clean_bgr is None:
            raise ValueError(f"Could not load clean image: {sample['clean_path']}")
            
        # Ensure dimensions match
        if noisy_bgr.shape != clean_bgr.shape:
            h, w = clean_bgr.shape[:2]
            noisy_bgr = noisy_bgr[:h, :w]
            
        if self.noise_level == "mixed":
            return {"noisy": noisy_bgr, "clean": clean_bgr, "name": sample["name"], "noise_level": sample["noise_level"]}
        else:
            return {"noisy": noisy_bgr, "clean": clean_bgr, "name": sample["name"]}
