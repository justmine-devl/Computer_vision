from src.restoration.wmgf_derain import WMGFConfig

def sample_wmgf_config(trial):
    return WMGFConfig(
        noise_window_size=3,
        window_small=3,
        window_medium=trial.suggest_categorical("window_medium", [5, 7]),
        window_large=trial.suggest_categorical("window_large", [7, 9]),
        noise_threshold_scale=trial.suggest_float("noise_threshold_scale", 0.5, 2.0),
        spatial_sigma=trial.suggest_float("spatial_sigma", 1.0, 5.0),
        range_sigma=trial.suggest_float("range_sigma", 0.05, 0.30),
        guided_radius=trial.suggest_int("guided_radius", 4, 32),
        guided_eps=trial.suggest_float("guided_eps", 1e-4, 1e-1, log=True),
        alpha=trial.suggest_float("alpha", 0.6, 1.0),
        gamma=trial.suggest_float("gamma", 0.8, 1.2),
        clahe_clip=trial.suggest_float("clahe_clip", 0.0, 2.0),
    )
