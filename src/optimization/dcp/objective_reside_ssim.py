import numpy as np
from src.optimization.dcp.search_space import sample_dcp_config
from src.restoration.dcp import DCPDehazeFilter
from src.metrics.full_reference import compute_ssim

class ObjectiveResideSSIM:
    def __init__(self, dataset):
        self.dataset = dataset

    def __call__(self, trial):
        config = sample_dcp_config(trial)
        dcp = DCPDehazeFilter(config)

        scores = []
        for sample in self.dataset:
            restored = dcp.restore(sample["hazy"])
            ssim = compute_ssim(restored, sample["clear"])
            scores.append(ssim)

        return float(np.mean(scores))
