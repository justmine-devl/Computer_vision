import numpy as np
from src.optimization.desnowing.search_space import sample_desnow_config
from src.restoration.morph_guided_desnow import MorphGuidedDesnowFilter
from src.metrics.no_reference import compute_brisque

class ObjectiveDawnBrisque:
    def __init__(self, dataset):
        self.dataset = dataset

    def __call__(self, trial):
        config = sample_desnow_config(trial)
        desnow = MorphGuidedDesnowFilter(config)

        scores = []
        for sample in self.dataset:
            # DAWN dataset yields "image"
            restored = desnow.restore(sample["image"])
            brisque = compute_brisque(restored)
            if not np.isnan(brisque):
                scores.append(brisque)

        if not scores:
            return float('inf')
        return float(np.mean(scores))
