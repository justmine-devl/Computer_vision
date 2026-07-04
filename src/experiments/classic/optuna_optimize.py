import os
import sys
import yaml
import json
import csv
import argparse
import numpy as np
import cv2
from pathlib import Path

def get_project_root():
    return Path(__file__).resolve().parents[3]

sys.path.append(str(get_project_root()))

from src.datasets.bsd_denoise_dataset import BSDDenoiseDataset
from src.datasets.reside_dataset import ResideDataset
from src.datasets.dawn_dataset import DawnDataset
from src.datasets.lol_dataset import LOLDataset
from src.datasets.rain100H_dataset import PairedRainDataset
from src.datasets.snow100k_dataset import Snow100kDataset
from src.datasets.gopro_dataset import GoProDataset

from src.optimization.bm3d.objective_bm3d_config1_specialized_ssim import objective_config1_specialized_ssim
from src.optimization.bm3d.objective_bm3d_config2_specialized_brisque import objective_config2_specialized_brisque
from src.optimization.bm3d.objective_bm3d_config1_mixed_ssim import objective_config1_mixed_ssim
from src.optimization.bm3d.objective_bm3d_config2_mixed_brisque import objective_config2_mixed_brisque

from src.optimization.dcp.objective_reside_ssim import ObjectiveResideSSIM
from src.optimization.dcp.objective_dawn_brisque import ObjectiveDawnBrisque as DCPObjectiveDawnBrisque
from src.optimization.dcp.objective_dawn_map50 import ObjectiveDawnMap50 as DCPObjectiveDawnMap50

from src.optimization.desnowing.objective_snow100k_ssim import ObjectiveSnow100KSSIM
from src.optimization.desnowing.objective_dawn_brisque import ObjectiveDawnBrisque as DesnowObjectiveDawnBrisque
from src.optimization.desnowing.objective_dawn_map50 import ObjectiveDawnMap50 as DesnowObjectiveDawnMap50

from src.optimization.rbcp.objective_desand_config2_brisque import objective_desand_config2_brisque
from src.optimization.rbcp.objective_desand_config3_map50 import objective_desand_config3_map50
from src.restoration.rbcp_desand import RBCPDesandFilter

from src.optimization.wmgf.objective_wmgf_config1_ssim import ObjectiveWMGFConfig1SSIM
from src.optimization.wmgf.objective_wmgf_config2_brisque import ObjectiveWMGFConfig2Brisque
from src.optimization.wmgf.objective_wmgf_config3_map50 import ObjectiveWMGFConfig3Map50
from src.detection.yolo_runner import YOLOEvaluator

from src.restoration.lime_delowlight import LIMEDeLowlightConfig, LIMEDeLowlightFilter
from src.metrics.full_reference import compute_ssim
from src.metrics.no_reference import compute_brisque
from src.restoration.wiener_deblur import WienerMotionDeblurFilter
from src.restoration.richardson_lucy_deblur import RichardsonLucyMotionDeblurFilter
from src.restoration.fergus_blind_deblur import FergusBlindMotionDeblurFilter

def sample_lime_config(trial):
    return LIMEDeLowlightConfig(
        illumination_floor=trial.suggest_float("illumination_floor", 0.01, 0.12),
        illumination_power=trial.suggest_float("illumination_power", 0.45, 1.10),
        guided_radius=trial.suggest_int("guided_radius", 4, 48),
        guided_eps=trial.suggest_float("guided_eps", 1e-5, 1e-1, log=True),
        exposure_gain=trial.suggest_float("exposure_gain", 0.8, 1.8),
        gamma=trial.suggest_float("gamma", 0.7, 1.8),
        blend_alpha=trial.suggest_float("blend_alpha", 0.5, 1.0),
        use_clahe=trial.suggest_categorical("use_clahe", [False, True]),
        clahe_clip=trial.suggest_float("clahe_clip", 0.0, 3.0),
        clahe_tile_grid_size=trial.suggest_categorical("clahe_tile_grid_size", [4, 8, 12]),
        use_denoise=trial.suggest_categorical("use_denoise", [False, True]),
        denoise_h=trial.suggest_float("denoise_h", 1.0, 8.0),
        denoise_h_color=trial.suggest_float("denoise_h_color", 1.0, 8.0),
        denoise_template_window=trial.suggest_categorical("denoise_template_window", [5, 7]),
        denoise_search_window=trial.suggest_categorical("denoise_search_window", [15, 21, 31]),
    )

def get_lime_ssim_objective(dataset):
    def objective(trial):
        config = sample_lime_config(trial)
        lime_filter = LIMEDeLowlightFilter(config)
        scores = []
        for low_img, high_gt in dataset:
            restored = lime_filter.restore(low_img)
            score = compute_ssim(restored, high_gt)
            scores.append(score)
        return np.mean(scores) if scores else 0.0
    return objective

def get_lime_brisque_objective(dataset):
    def objective(trial):
        config = sample_lime_config(trial)
        lime_filter = LIMEDeLowlightFilter(config)
        scores = []
        for low_img in dataset:
            restored = lime_filter.restore(low_img)
            score = compute_brisque(restored)
            if not np.isnan(score):
                scores.append(score)
        return np.mean(scores) if scores else float('inf')
    return objective

def get_motion_deblur_objective(dataset, metric="ssim"):
    def objective(trial):
        method = trial.suggest_categorical("method", ["wiener", "richardson_lucy", "fergus_blind"])
        
        if method == "wiener":
            kernel_length = trial.suggest_int("kernel_length", 3, 45, step=2)
            kernel_angle = trial.suggest_float("kernel_angle", 0.0, 180.0)
            reg_lambda = trial.suggest_float("regularization_lambda", 1e-6, 1e-1, log=True)
            post_sharpen_amount = trial.suggest_float("post_sharpen_amount", 0.0, 1.0)
            
            cfg = {
                'kernel_length': kernel_length,
                'kernel_angle': kernel_angle,
                'regularization_lambda': reg_lambda,
                'post_sharpen': True if post_sharpen_amount > 0 else False,
                'post_sharpen_amount': post_sharpen_amount,
                'clip_output': True
            }
            filt = WienerMotionDeblurFilter(cfg)
            
        elif method == "richardson_lucy":
            kernel_length = trial.suggest_int("kernel_length", 3, 45, step=2)
            kernel_angle = trial.suggest_float("kernel_angle", 0.0, 180.0)
            num_iter = trial.suggest_int("num_iter", 5, 80)
            denoise_after = trial.suggest_categorical("denoise_after", [False, True])
            
            cfg = {
                'kernel_length': kernel_length,
                'kernel_angle': kernel_angle,
                'num_iter': num_iter,
                'denoise_after': denoise_after,
                'clip_output': True
            }
            filt = RichardsonLucyMotionDeblurFilter(cfg)
            
        elif method == "fergus_blind":
            kernel_length = trial.suggest_int("kernel_length", 5, 65, step=2)
            kernel_angle = trial.suggest_float("kernel_angle", 0.0, 180.0)
            deconv_method = trial.suggest_categorical("deconv_method", ["wiener", "richardson_lucy"])
            reg_lambda = trial.suggest_float("regularization_lambda", 1e-6, 1e-1, log=True)
            num_iter = trial.suggest_int("num_iter", 10, 80)
            
            cfg = {
                'per_image_search': False,
                'kernel_length': kernel_length,
                'kernel_angle': kernel_angle,
                'deconv_method': deconv_method,
                'regularization_lambda': reg_lambda,
                'num_iter': num_iter,
                'clip_output': True
            }
            filt = FergusBlindMotionDeblurFilter(cfg)
            
        scores = []
        subset_size = min(50, len(dataset))
        for i in range(subset_size):
            data = dataset[i]
            restored = filt.apply(data['blur'])
            if metric == "ssim":
                score = compute_ssim(restored, data['sharp'])
                scores.append(score)
            else: # brisque
                score = compute_brisque(restored)
                if not np.isnan(score):
                    scores.append(score)
                    
        if not scores:
            return float('inf') if metric == "brisque" else 0.0
        return np.mean(scores)
    return objective

def main():
    parser = argparse.ArgumentParser(description="Unified Optuna optimization runner for ProjectCV.")
    parser.add_argument(
        "--project-root",
        type=str,
        default=str(get_project_root()),
        help="Path to the root directory of the project."
    )
    parser.add_argument(
        "--yolo-weights",
        type=str,
        default="yolo26n.pt",
        help="Filename or path of YOLO weights located relative to project root."
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["bm3d", "dcp", "lime", "wmgf", "desnow", "rbcp", "motion"],
        help="The restoration method to optimize."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="The specific experiment config name (e.g., config1_reside_ssim, config1_ssim, specialized_ssim, etc.)"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Number of trials for Optuna optimization."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/classic/optuna",
        help="Directory for Optuna trial outputs."
    )
    args = parser.parse_args()
    import optuna

    root_path = Path(args.project_root)
    method = args.method
    config_name = args.config
    n_trials = args.n_trials
    
    # Resolve YOLO weights path
    yolo_weights_path = Path(args.yolo_weights)
    if not yolo_weights_path.is_absolute():
        yolo_weights_path = root_path / yolo_weights_path
    
    print(f"Project Root: {root_path}")
    print(f"Using YOLO Weights: {yolo_weights_path}")
    print(f"Optimizing method: {method} with config: {config_name} (trials={n_trials})")
    
    # Ensure folders exist
    configs_dir = root_path / "configs" / method
    configs_dir.mkdir(parents=True, exist_ok=True)
    
    optuna_results_dir = Path(args.output_dir)
    if not optuna_results_dir.is_absolute():
        optuna_results_dir = root_path / optuna_results_dir
    optuna_results_dir = optuna_results_dir / f"{method}_{config_name}"
    optuna_results_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. BM3D Optimization logic
    if method == "bm3d":
        noise_levels = [15, 25, 50]
        if "specialized" in config_name:
            direction = "minimize" if "brisque" in config_name else "maximize"
            for level in noise_levels:
                print(f"--- Optimizing BM3D Specialized config for Noise Level {level} ---")
                val_dataset = BSDDenoiseDataset(root=str(root_path / 'data' / 'bsd_denoise'), noise_level=level, split='val')
                
                study = optuna.create_study(direction=direction)
                if "ssim" in config_name:
                    study.optimize(lambda trial: objective_config1_specialized_ssim(trial, level, val_dataset), n_trials=n_trials)
                else:
                    study.optimize(lambda trial: objective_config2_specialized_brisque(trial, level, val_dataset), n_trials=n_trials)
                
                # Save trials
                study.trials_dataframe().to_csv(optuna_results_dir / f"bm3d_{config_name}_level{level}_trials.csv", index=False)
                
                # Save config
                best_params = study.best_params
                config_data = {
                    'name': f'bm3d_{config_name}_level{level}',
                    'sigma_psd': best_params['sigma_psd'],
                    'stage_arg': best_params['stage_arg'],
                    'profile': 'default'
                }
                yaml_cfg_path = root_path / "configs" / "bm3d" / f"bm3d_{config_name.replace('specialized', f'bsd{level}')}.yaml"
                with open(yaml_cfg_path, 'w') as f:
                    yaml.dump(config_data, f)
                print(f"Saved config to {yaml_cfg_path}. Best params: {best_params}")
                
        elif "mixed" in config_name:
            direction = "minimize" if "brisque" in config_name else "maximize"
            val_datasets_dict = {
                level: BSDDenoiseDataset(root=str(root_path / 'data' / 'bsd_denoise'), noise_level=level, split='val')
                for level in noise_levels
            }
            study = optuna.create_study(direction=direction)
            if "ssim" in config_name:
                study.optimize(lambda trial: objective_config1_mixed_ssim(trial, val_datasets_dict), n_trials=n_trials)
            else:
                study.optimize(lambda trial: objective_config2_mixed_brisque(trial, val_datasets_dict), n_trials=n_trials)
                
            study.trials_dataframe().to_csv(optuna_results_dir / f"bm3d_{config_name}_trials.csv", index=False)
            
            best_params = study.best_params
            config_data = {
                'name': f'bm3d_{config_name}',
                'sigma_psd': best_params['sigma_psd'],
                'stage_arg': best_params['stage_arg'],
                'profile': 'default'
            }
            yaml_cfg_path = root_path / "configs" / "bm3d" / f"bm3d_{config_name.replace('mixed', 'bsd_mixed')}.yaml"
            with open(yaml_cfg_path, 'w') as f:
                yaml.dump(config_data, f)
            print(f"Saved mixed config to {yaml_cfg_path}. Best params: {best_params}")
            
    # 2. DCP Optimization logic
    elif method == "dcp":
        if "reside_ssim" in config_name:
            val_file = root_path / "data" / "reside6k" / "splits" / "val.txt"
            dataset = ResideDataset(str(val_file), str(root_path / "dataset" / "RESIDE-6K"))
            objective = ObjectiveResideSSIM(dataset)
            direction = "maximize"
        elif "dawn_brisque" in config_name:
            val_csv = root_path / "data" / "dawn_fog" / "splits" / "fog_val_pairs.csv"
            dataset = DawnDataset(str(val_csv), "")
            objective = DCPObjectiveDawnBrisque(dataset)
            direction = "minimize"
        elif "dawn_map50" in config_name:
            val_csv = root_path / "data" / "dawn_fog" / "splits" / "fog_val_pairs.csv"
            dataset = DawnDataset(str(val_csv), "")
            yolo_eval = YOLOEvaluator(str(yolo_weights_path), "")
            objective = DCPObjectiveDawnMap50(dataset, yolo_eval, str(optuna_results_dir))
            direction = "maximize"
        else:
            raise ValueError(f"Unknown config for DCP: {config_name}")
            
        study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials)
        
        # Save trials
        study.trials_dataframe().to_csv(optuna_results_dir / "trials.csv", index=False)
        # Save best config in output folder
        with open(optuna_results_dir / "best_config.yaml", "w") as f:
            yaml.dump(study.best_params, f)
        # Save yaml configuration into configs/
        with open(root_path / "configs" / "dcp" / f"optuna_{config_name}.yaml", "w") as f:
            yaml.dump(study.best_params, f)
        print(f"DCP Optimization completed. Best value: {study.best_value}. Best params: {study.best_params}")
        
    # 3. LIME Optimization logic
    elif method == "lime":
        val_low_dir = root_path / "data" / "lol" / "val" / "low"
        val_high_dir = root_path / "data" / "lol" / "val" / "high"
        
        low_images = sorted(list(val_low_dir.glob("*.png")))
        
        if "ssim" in config_name:
            dataset = []
            for low_path in low_images:
                high_path = val_high_dir / low_path.name
                if high_path.exists():
                    dataset.append((cv2.imread(str(low_path)), cv2.imread(str(high_path))))
            objective = get_lime_ssim_objective(dataset)
            direction = "maximize"
        elif "brisque" in config_name:
            dataset = [cv2.imread(str(low_path)) for low_path in low_images]
            objective = get_lime_brisque_objective(dataset)
            direction = "minimize"
        else:
            raise ValueError(f"Unknown config for LIME: {config_name}")
            
        study = optuna.create_study(direction=direction)
        study.optimize(objective, n_trials=n_trials)
        
        best_params = study.best_params.copy()
        best_config_obj = LIMEDeLowlightConfig(**best_params)
        final_dict = {
            "name": f"lime_delowlight_{config_name}",
            "illumination_floor": best_config_obj.illumination_floor,
            "illumination_power": best_config_obj.illumination_power,
            "guided_radius": best_config_obj.guided_radius,
            "guided_eps": best_config_obj.guided_eps,
            "exposure_gain": best_config_obj.exposure_gain,
            "gamma": best_config_obj.gamma,
            "blend_alpha": best_config_obj.blend_alpha,
            "use_clahe": best_config_obj.use_clahe,
            "clahe_clip": best_config_obj.clahe_clip,
            "clahe_tile_grid_size": best_config_obj.clahe_tile_grid_size,
            "use_denoise": best_config_obj.use_denoise,
            "denoise_h": best_config_obj.denoise_h,
            "denoise_h_color": best_config_obj.denoise_h_color,
            "denoise_template_window": best_config_obj.denoise_template_window,
            "denoise_search_window": best_config_obj.denoise_search_window,
        }
        
        # Save trials
        study.trials_dataframe().to_csv(optuna_results_dir / "trials.csv", index=False)
        # Save best config in output folder
        with open(optuna_results_dir / "best_config.yaml", "w") as f:
            yaml.dump(final_dict, f, default_flow_style=False, sort_keys=False)
        # Save to configs/
        with open(root_path / "configs" / "lime" / f"lime_delowlight_{config_name}.yaml", "w") as f:
            yaml.dump(final_dict, f, default_flow_style=False, sort_keys=False)
        print(f"LIME Optimization completed. Best value: {study.best_value}. Best params: {study.best_params}")

    # 4. WMGF Optimization logic
    elif method == "wmgf":
        if "ssim" in config_name:
            val_file = root_path / "data" / "rain100h" / "splits" / "val.txt"
            dataset = PairedRainDataset(str(val_file), str(root_path / "dataset" / "rain100H"))
            objective = ObjectiveWMGFConfig1SSIM(dataset)
            direction = "maximize"
        elif "brisque" in config_name:
            val_csv = root_path / "data" / "dawn_rain" / "splits" / "rain_val_pairs.csv"
            dataset = DawnDataset(str(val_csv), "")
            objective = ObjectiveWMGFConfig2Brisque(dataset)
            direction = "minimize"
        elif "map50" in config_name:
            val_csv = root_path / "data" / "dawn_rain" / "splits" / "rain_val_pairs.csv"
            dataset = DawnDataset(str(val_csv), "")
            yolo_eval = YOLOEvaluator(str(yolo_weights_path), "")
            objective = ObjectiveWMGFConfig3Map50(dataset, yolo_eval, str(optuna_results_dir))
            direction = "maximize"
        else:
            raise ValueError(f"Unknown config for WMGF: {config_name}")
            
        study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials)
        
        study.trials_dataframe().to_csv(optuna_results_dir / "trials.csv", index=False)
        with open(optuna_results_dir / "best_config.yaml", "w") as f:
            yaml.dump(study.best_params, f)
        with open(root_path / "configs" / "wmgf" / f"derain_optuna_{config_name}.yaml", "w") as f:
            yaml.dump(study.best_params, f)
        print(f"WMGF Optimization completed. Best value: {study.best_value}. Best params: {study.best_params}")

    # 5. Desnow Optimization logic
    elif method == "desnow":
        if "ssim" in config_name:
            val_file = root_path / "data" / "snow100k" / "splits" / "val.txt"
            dataset = Snow100kDataset(str(val_file), str(root_path / "dataset" / "Snow100K"))
            objective = ObjectiveSnow100KSSIM(dataset)
            direction = "maximize"
        elif "brisque" in config_name:
            val_csv = root_path / "data" / "dawn" / "splits" / "snow_val_pairs.csv"
            dataset = DawnDataset(str(val_csv), "")
            objective = DesnowObjectiveDawnBrisque(dataset)
            direction = "minimize"
        elif "map50" in config_name:
            val_csv = root_path / "data" / "dawn" / "splits" / "snow_val_pairs.csv"
            dataset = DawnDataset(str(val_csv), "")
            yolo_eval = YOLOEvaluator(str(yolo_weights_path), "")
            objective = DesnowObjectiveDawnMap50(dataset, yolo_eval, str(optuna_results_dir))
            direction = "maximize"
        else:
            raise ValueError(f"Unknown config for Desnow: {config_name}")
            
        study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials)
        
        study.trials_dataframe().to_csv(optuna_results_dir / "trials.csv", index=False)
        with open(optuna_results_dir / "best_config.yaml", "w") as f:
            yaml.dump(study.best_params, f)
        with open(root_path / "configs" / "desnowing" / f"optuna_{config_name}.yaml", "w") as f:
            yaml.dump(study.best_params, f)
        print(f"Desnow Optimization completed. Best value: {study.best_value}. Best params: {study.best_params}")

    # 6. RBCP Optimization logic
    elif method == "rbcp":
        val_csv = root_path / "data" / "dawn" / "splits" / "sand_val_pairs.csv"
        if not val_csv.exists():
            print(f"Please run prepare_data.py for dawn_sand first. Missing split: {val_csv}")
            return
            
        pairs = []
        with open(val_csv, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) == 2:
                    pairs.append((row[0], row[1]))
                    
        if "brisque" in config_name:
            objective = lambda trial: objective_desand_config2_brisque(trial, RBCPDesandFilter, pairs)
            direction = "minimize"
        elif "map50" in config_name:
            dataset_yaml = root_path / "data" / "dawn" / "dawn_sand.yaml"
            objective = lambda trial: objective_desand_config3_map50(trial, RBCPDesandFilter, str(dataset_yaml), pairs, yolo_model_path=str(yolo_weights_path))
            direction = "maximize"
        else:
            raise ValueError(f"Unknown config for RBCP: {config_name}")
            
        study = optuna.create_study(direction=direction)
        study.optimize(objective, n_trials=n_trials)
        
        # Save trials
        with open(optuna_results_dir / "trials.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["trial_id", "value", "state"] + list(study.best_params.keys()))
            for t in study.trials:
                row = [t.number, t.value, t.state.name]
                for k in study.best_params.keys():
                    row.append(t.params.get(k, ""))
                writer.writerow(row)
                
        best_config = study.best_params.copy()
        best_config['use_guided_filter'] = True
        
        with open(optuna_results_dir / "best_config.yaml", "w") as f:
            yaml.dump(best_config, f)
            
        with open(root_path / "configs" / "rbcp" / f"desand_optuna_{config_name}.yaml", "w") as f:
            yaml.dump(best_config, f)
            
        print(f"RBCP Optimization completed. Best value: {study.best_value}. Best params: {study.best_params}")

    # 7. Motion deblur Optimization logic
    elif method == "motion":
        csv_file = root_path / "data" / "gopro" / "gopro_pairs.csv"
        if not csv_file.exists():
            print(f"Please run prepare_data.py first. Missing index: {csv_file}")
            return
            
        dataset = GoProDataset(str(csv_file), split='val')
        
        if "ssim" in config_name:
            objective = get_motion_deblur_objective(dataset, "ssim")
            direction = "maximize"
        elif "brisque" in config_name:
            objective = get_motion_deblur_objective(dataset, "brisque")
            direction = "minimize"
        else:
            raise ValueError(f"Unknown config for Motion Deblur: {config_name}")
            
        study = optuna.create_study(direction=direction, study_name=f"GoPro_{config_name}")
        study.optimize(objective, n_trials=n_trials)
        
        study.trials_dataframe().to_csv(optuna_results_dir / "trials.csv", index=False)
        with open(optuna_results_dir / "best_config.json", "w") as f:
            json.dump(study.best_params, f, indent=4)
            
        # We also dump YAML config format for consistency if needed
        with open(root_path / "configs" / "motion_deblur" / f"optuna_{config_name}.yaml", "w") as f:
            yaml.dump(study.best_params, f)
            
        print(f"Motion Deblur Optimization completed. Best value: {study.best_value}. Best params: {study.best_params}")

if __name__ == "__main__":
    main()
