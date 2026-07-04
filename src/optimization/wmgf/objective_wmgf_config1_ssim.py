import numpy as np
from src.optimization.wmgf.search_space import sample_wmgf_config
from src.restoration.wmgf_derain import WMGFDerainFilter
from src.metrics.full_reference import compute_ssim

class ObjectiveWMGFConfig1SSIM:
    def __init__(self, dataset):
        self.dataset = dataset

    def __call__(self, trial):
        config = sample_wmgf_config(trial)
        filter_wmgf = WMGFDerainFilter(config)

        scores = []
        for sample in self.dataset:
            restored = filter_wmgf.restore(sample["rainy"])
            ssim = compute_ssim(restored, sample["clean"])
            scores.append(ssim)

        return float(np.mean(scores))
