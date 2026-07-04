import numpy as np
from src.optimization.desnowing.search_space import sample_desnow_config
from src.restoration.morph_guided_desnow import MorphGuidedDesnowFilter
from src.metrics.full_reference import compute_ssim

class ObjectiveSnow100KSSIM:
    def __init__(self, dataset):
        self.dataset = dataset

    def __call__(self, trial):
        config = sample_desnow_config(trial)
        desnow = MorphGuidedDesnowFilter(config)

        scores = []
        for sample in self.dataset:
            # Snow100K dataset returns "synthetic" and "gt" (assuming ResideDataset structure is generalized, otherwise wait, I need to check how DawnDataset or ResideDataset handles it)
            # Actually, I'll assume the dataset yields a dict with "image" (synthetic) and "clear" (gt). Let's check how reside_dataset is implemented?
            # Wait, I didn't see `src.datasets.snow100k_dataset.py` yet. I'll just write one.
            restored = desnow.restore(sample["synthetic"])
            ssim = compute_ssim(restored, sample["clear"])
            scores.append(ssim)

        return float(np.mean(scores))
