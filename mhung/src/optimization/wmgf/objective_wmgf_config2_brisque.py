import numpy as np
from src.optimization.wmgf.search_space import sample_wmgf_config
from src.restoration.wmgf_derain import WMGFDerainFilter
from src.metrics.no_reference import compute_brisque

class ObjectiveWMGFConfig2Brisque:
    def __init__(self, dataset):
        self.dataset = dataset

    def __call__(self, trial):
        config = sample_wmgf_config(trial)
        filter_wmgf = WMGFDerainFilter(config)

        scores = []
        for sample in self.dataset:
            restored = filter_wmgf.restore(sample["image"])
            brisque = compute_brisque(restored)
            scores.append(brisque)

        return float(np.mean(scores))
