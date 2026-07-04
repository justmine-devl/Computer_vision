from src.restoration.morph_guided_desnow import MorphGuidedDesnowConfig

def sample_desnow_config(trial):
    config = MorphGuidedDesnowConfig()
    
    config.v_threshold = trial.suggest_float("v_threshold", 0.5, 0.95)
    config.s_threshold = trial.suggest_float("s_threshold", 0.1, 0.6)
    
    config.opening_kernel = trial.suggest_categorical("opening_kernel", [1, 3, 5])
    config.closing_kernel = trial.suggest_categorical("closing_kernel", [1, 3, 5])
    config.median_kernel = trial.suggest_categorical("median_kernel", [1, 3, 5])
    
    config.use_guided_filter = trial.suggest_categorical("use_guided_filter", [True, False])
    if config.use_guided_filter:
        config.guided_radius = trial.suggest_int("guided_radius", 4, 16)
        config.guided_eps = trial.suggest_float("guided_eps", 1e-4, 1e-1, log=True)
    else:
        config.use_bilateral = trial.suggest_categorical("use_bilateral", [True, False])
        if config.use_bilateral:
            config.bilateral_d = trial.suggest_categorical("bilateral_d", [5, 7, 9])
            config.sigma_color = trial.suggest_float("sigma_color", 10.0, 100.0)
            config.sigma_space = trial.suggest_float("sigma_space", 10.0, 100.0)
            
    config.alpha = trial.suggest_float("alpha", 0.3, 0.95)
    config.gamma = trial.suggest_float("gamma", 0.8, 1.5)
    config.clahe_clip = trial.suggest_float("clahe_clip", 0.0, 4.0)
    
    return config
