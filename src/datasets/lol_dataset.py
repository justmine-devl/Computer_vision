import cv2
from pathlib import Path

class LOLDataset:
    def __init__(self, split_file=None, base_dir=None):
        if base_dir is None:
            raise ValueError("base_dir is required. Example: --data-root data/lol")
        self.base_dir = base_dir
        
        low_dir = Path(self.base_dir) / "eval15" / "low"
        high_dir = Path(self.base_dir) / "eval15" / "high"
        
        self.samples = []
        if low_dir.exists() and high_dir.exists():
            for low_path in sorted(low_dir.glob("*.png")):
                high_path = high_dir / low_path.name
                if high_path.exists():
                    self.samples.append({
                        "low": str(low_path),
                        "normal": str(high_path),
                        "low_rel": low_path.name
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        low_img = cv2.imread(sample["low"])
        normal_img = cv2.imread(sample["normal"])
        
        return {
            "low": low_img,
            "normal": normal_img,
            "image_path": sample["low"],
            "image_rel": sample["low_rel"]
        }

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
