import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.restoration.dcp import DCPConfig

def sample_dcp_config(trial) -> DCPConfig:
    return DCPConfig(
        patch_size=trial.suggest_categorical("patch_size", [7, 11, 15, 21, 31]),
        omega=trial.suggest_float("omega", 0.75, 0.98),
        t0=trial.suggest_float("t0", 0.05, 0.35),
        top_percent=trial.suggest_float("top_percent", 0.0005, 0.01, log=True),
        guided_radius=trial.suggest_int("guided_radius", 20, 80),
        guided_eps=trial.suggest_float("guided_eps", 1e-4, 1e-1, log=True),
        gamma=trial.suggest_float("gamma", 0.8, 1.3),
        clahe_clip=trial.suggest_float("clahe_clip", 0.0, 3.0),
    )
