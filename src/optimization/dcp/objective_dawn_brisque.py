import numpy as np
from src.optimization.dcp.search_space import sample_dcp_config
from src.restoration.dcp import DCPDehazeFilter
from src.metrics.no_reference import compute_brisque

class ObjectiveDawnBrisque:
    def __init__(self, dataset):
        self.dataset = dataset

    def __call__(self, trial):
        config = sample_dcp_config(trial)
        dcp = DCPDehazeFilter(config)

        scores = []
        for sample in self.dataset:
            restored = dcp.restore(sample["image"])
            brisque = compute_brisque(restored)
            scores.append(brisque)

        return float(np.mean(scores))
