import os
import cv2

class Snow100kDataset:
    def __init__(self, split_file, base_dir):
        self.base_dir = base_dir
        self.samples = []
        with open(split_file, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) == 2:
                    syn_rel, gt_rel = parts
                    self.samples.append({
                        "synthetic": os.path.join(base_dir, syn_rel),
                        "clear": os.path.join(base_dir, gt_rel)
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        syn_img = cv2.imread(sample["synthetic"])
        clear_img = cv2.imread(sample["clear"])
        return {
            "synthetic": syn_img,
            "clear": clear_img,
            "syn_path": sample["synthetic"]
        }

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
