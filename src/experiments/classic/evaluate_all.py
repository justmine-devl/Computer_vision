import os
import sys
import yaml
import numpy as np
import pandas as pd
import cv2
import shutil
import argparse
from pathlib import Path

def get_project_root():
    return Path(__file__).resolve().parents[3]

ROOT = get_project_root()
sys.path.insert(0, str(ROOT))

from src.restoration.registry import create_filter
from src.datasets.reside_dataset import ResideDataset
from src.datasets.dawn_dataset import DawnDataset
from src.datasets.rain100H_dataset import PairedRainDataset
from src.datasets.snow100k_dataset import Snow100kDataset
from src.datasets.bsd_denoise_dataset import BSDDenoiseDataset
from src.datasets.lol_dataset import LOLDataset
from src.datasets.gopro_dataset import GoProDataset
from src.datasets.raindrop_dataset import RainDropDataset
from src.metrics.full_reference import compute_ssim, compute_psnr
from src.metrics.no_reference import compute_brisque, compute_niqe
from src.detection.yolo_runner import YOLOEvaluator
from src.metrics.detection import compute_mean_iou

def safe_compute(func, *args, **kwargs):
    """Safely compute a metric function, catching exceptions and wrapping NaN values."""
    try:
        val = func(*args, **kwargs)
        if np.isnan(val) or np.isinf(val): 
            return None
        return val
    except Exception:
        return None

def get_basename(sample):
    """Retrieve the basename of a sample dynamically from its paths."""
    if "image_rel" in sample: 
        return sample["image_rel"]
    for key in ["hazy_path", "blur_path", "low_path", "rainy_path", "image_path"]:
        if key in sample: 
            return os.path.basename(sample[key])
    return str(hash(str(sample.keys()))) + ".jpg"

def cache_dataset(filter_obj, dataset, img_key, cache_dir, quick_test=False):
    """Process images with the given filter and cache the output restored images."""
    os.makedirs(cache_dir, exist_ok=True)
    count = 0
    for sample in dataset:
        if quick_test and count >= 2:
            break
        basename = get_basename(sample)
        img_path = os.path.join(cache_dir, basename)
        if not os.path.exists(img_path):
            restored = filter_obj.restore(sample[img_key]) if filter_obj is not None else sample[img_key]
            # Ensure output dir exists for relative folder structured basenames
            os.makedirs(os.path.dirname(img_path), exist_ok=True)
            cv2.imwrite(img_path, restored)
        count += 1

def eval_fr(dataset, gt_key, cache_dir, quick_test=False):
    """Compute Full-Reference metrics (SSIM, PSNR) on cached images."""
    ssims, psnrs = [], []
    count = 0
    for sample in dataset:
        if quick_test and count >= 2:
            break
        basename = get_basename(sample)
        img_path = os.path.join(cache_dir, basename)
        if os.path.exists(img_path):
            restored = cv2.imread(img_path)
            gt = sample[gt_key]
            s = safe_compute(compute_ssim, restored, gt)
            p = safe_compute(compute_psnr, restored, gt)
            if s is not None: 
                ssims.append(s)
            if p is not None: 
                psnrs.append(p)
        count += 1
    return np.mean(ssims) if ssims else 0.0, np.mean(psnrs) if psnrs else 0.0

def eval_nr(dataset, cache_dir, quick_test=False):
    """Compute No-Reference metrics (BRISQUE, NIQE) on cached images."""
    brisques, niqes = [], []
    count = 0
    for sample in dataset:
        if quick_test and count >= 2:
            break
        basename = get_basename(sample)
        img_path = os.path.join(cache_dir, basename)
        if os.path.exists(img_path):
            restored = cv2.imread(img_path)
            b = safe_compute(compute_brisque, restored)
            n = safe_compute(compute_niqe, restored)
            if b is not None: 
                brisques.append(b)
            if n is not None: 
                niqes.append(n)
        count += 1
    return np.mean(brisques) if brisques else 0.0, np.mean(niqes) if niqes else 0.0

def eval_yolo(dataset, results_dir, yolo_weights_path, dataset_name="", quick_test=False):
    """Evaluate YOLO detection performance on restored images."""
    from ultralytics import YOLO

    images_dir = os.path.join(results_dir, "images")
    labels_dir = os.path.join(results_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    count = 0
    for sample in dataset:
        if quick_test and count >= 2:
            break
        if sample.get("label_path") and os.path.exists(sample["label_path"]):
            dst_label = os.path.join(labels_dir, sample["label_rel"])
            os.makedirs(os.path.dirname(dst_label), exist_ok=True)
            if "FoggyCityscapes" in dataset_name.replace(" ", ""):
                # Convert Foggy Cityscapes categories to standard COCO indices
                with open(sample["label_path"], "r") as src_f, open(dst_label, "w") as dst_f:
                    for line in src_f:
                        parts = line.strip().split()
                        if not parts: 
                            continue
                        cls_id = int(parts[0])
                        # Map 0: Car -> 2, 1: Person -> 0
                        if cls_id == 0: 
                            cls_id = 2
                        elif cls_id == 1: 
                            cls_id = 0
                        parts[0] = str(cls_id)
                        dst_f.write(" ".join(parts) + "\n")
            else:
                shutil.copy2(sample["label_path"], dst_label)
        count += 1
    
    yaml_path = os.path.join(results_dir, "temp.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {os.path.abspath(results_dir)}\n")
        f.write(f"train: images\nval: images\n")
        f.write("names:\n  0: person\n  1: bicycle\n  2: car\n  3: motorcycle\n  4: airplane\n  5: bus\n  6: train\n  7: truck\n")
    
    evaluator = YOLOEvaluator(yolo_weights_path, yaml_path)
    
    unique_classes = set()
    for root_dir, _, files in os.walk(labels_dir):
        for f in files:
            if f.endswith('.txt'):
                with open(os.path.join(root_dir, f), 'r') as txt:
                    for line in txt:
                        parts = line.strip().split()
                        if parts:
                            unique_classes.add(int(parts[0]))
    unique_classes = sorted(list(unique_classes))
    if not unique_classes:
        unique_classes = [0, 1, 2, 3, 5, 7]
        
    metrics = evaluator.evaluate(yaml_path, classes=unique_classes) or {}
    
    yolo_model = YOLO(yolo_weights_path)
    mean_iou = compute_mean_iou(yolo_model, images_dir, labels_dir, allowed_classes=unique_classes)
    metrics['mean_iou'] = mean_iou
    
    # Cleanup temp YOLO val file
    if os.path.exists(yaml_path):
        os.remove(yaml_path)
        
    return metrics

def safe_init_dataset(cls, *args, **kwargs):
    """Safely instantiate a dataset, returning None if split file or directory does not exist."""
    if len(args) > 0 and isinstance(args[0], str):
        if not os.path.exists(args[0]):
            return None
    try:
        return cls(*args, **kwargs)
    except Exception:
        return None

def run_evaluation():
    parser = argparse.ArgumentParser(description="Run evaluation on all filters for project reporting.")
    parser.add_argument(
        "--project-root",
        type=str,
        default=str(ROOT),
        help="Path to the project root directory."
    )
    parser.add_argument(
        "--yolo-weights",
        type=str,
        default="yolo26n.pt",
        help="Filename or path of YOLO weights located relative to project root."
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Enable quick test mode to process only 2 samples per configuration."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/classic/evaluate_all",
        help="Directory for raw evaluation outputs."
    )
    args = parser.parse_args()
    
    base_dir = Path(args.project_root)
    quick_test = args.quick_test
    
    # Resolve YOLO weights path
    yolo_weights_path = Path(args.yolo_weights)
    if not yolo_weights_path.is_absolute():
        yolo_weights_path = base_dir / yolo_weights_path
    yolo_weights_path = str(yolo_weights_path)
    
    print(f"Using Project Root: {base_dir}")
    print(f"Using YOLO Weights: {yolo_weights_path}")
    print(f"Quick Test Mode: {quick_test}")
    
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = base_dir / output_dir
    paper_tables_dir = output_dir / "tables"
    paper_figures_dir = output_dir / "figures" / "detection"
    paper_tables_dir.mkdir(parents=True, exist_ok=True)
    paper_figures_dir.mkdir(parents=True, exist_ok=True)
    
    configs = {
        "DCP_DAWN": {
            "type": "dcp", "img_fr": "hazy", "gt_fr": "clear",
            "configs": {
                "Raw": None,
                "Default": "configs/dcp/dcp_default.yaml",
                "Config1": "configs/dcp/optuna_config1_reside_ssim.yaml",
                "Config2": "configs/dcp/optuna_config2_dawn_brisque.yaml",
                "Config3": "configs/dcp/optuna_config3_dawn_map50.yaml"
            },
            "ds_fr": safe_init_dataset(ResideDataset, str(base_dir / "data/reside6k/splits/test.txt"), str(base_dir / "dataset/RESIDE-6K")),
            "ds_nr": safe_init_dataset(DawnDataset, str(base_dir / "data/dawn_fog/splits/fog_test_pairs.csv"), "")
        },
        "DCP_RTTS": {
            "type": "dcp", "img_fr": None, "gt_fr": None,
            "configs": {
                "Raw": None,
                "Default": "configs/dcp/dcp_default.yaml",
                "Config1": "configs/dcp/optuna_config1_reside_ssim.yaml",
                "Config2": "configs/dcp/optuna_config2_dawn_brisque.yaml",
                "Config3": "configs/dcp/optuna_config3_dawn_map50.yaml"
            },
            "ds_fr": None,
            "ds_nr": safe_init_dataset(DawnDataset, str(base_dir / "data/rtts/splits/val_test_pairs.csv"), "")
        },
        "DCP_FoggyCityscapes": {
            "type": "dcp", "img_fr": None, "gt_fr": None,
            "configs": {
                "Raw": None,
                "Default": "configs/dcp/dcp_default.yaml",
                "Config1": "configs/dcp/optuna_config1_reside_ssim.yaml",
                "Config2": "configs/dcp/optuna_config2_dawn_brisque.yaml",
                "Config3": "configs/dcp/optuna_config3_dawn_map50.yaml"
            },
            "ds_fr": None,
            "ds_nr": safe_init_dataset(DawnDataset, str(base_dir / "data/foggycityscapes/splits/val_test_pairs.csv"), "")
        },
        "WMGF": {
            "type": "wmgf", "img_fr": "rainy", "gt_fr": "clean",
            "configs": {
                "Raw": None,
                "Default": "configs/wmgf/derain_wmgf_default.yaml",
                "Config1": "configs/wmgf/derain_optuna_config1_ssim.yaml",
                "Config2": "configs/wmgf/derain_optuna_config2_brisque.yaml",
                "Config3": "configs/wmgf/derain_optuna_config3_map50.yaml"
            },
            "ds_fr": safe_init_dataset(PairedRainDataset, str(base_dir / "data/rain100h/splits/test.txt"), str(base_dir / "dataset/rain100H")),
            "ds_nr": safe_init_dataset(DawnDataset, str(base_dir / "data/dawn_rain/splits/rain_test_pairs.csv"), "")
        },
        "WMGF_RainDrop": {
            "type": "wmgf", "img_fr": "rainy", "gt_fr": "clean",
            "configs": {
                "Raw": None,
                "Default": "configs/wmgf/derain_wmgf_default.yaml",
                "Config1": "configs/wmgf/derain_optuna_config1_ssim.yaml",
                "Config2": "configs/wmgf/derain_optuna_config2_brisque.yaml",
                "Config3": "configs/wmgf/derain_optuna_config3_map50.yaml"
            },
            "ds_fr": safe_init_dataset(RainDropDataset, str(base_dir / "dataset/RainDrop")),
            "ds_nr": None
        },
        "Desnow": {
            "type": "desnow", "img_fr": "synthetic", "gt_fr": "clear",
            "configs": {
                "Raw": None,
                "Default": "configs/desnowing/desnow_morph_guided_default.yaml",
                "Config1": "configs/desnowing/optuna_config1_ssim.yaml",
                "Config2": "configs/desnowing/optuna_config2_brisque.yaml",
                "Config3": "configs/desnowing/optuna_config3_map50.yaml"
            },
            "ds_fr": safe_init_dataset(Snow100kDataset, str(base_dir / "data/snow100k/splits/test.txt"), str(base_dir / "dataset/Snow100K")),
            "ds_nr": safe_init_dataset(DawnDataset, str(base_dir / "data/dawn/splits/snow_test_pairs.csv"), "")
        },
        "LIME": {
            "type": "lime", "img_fr": "low", "gt_fr": "normal",
            "configs": {
                "Raw": None,
                "Default": "configs/lime/lime_delowlight_default.yaml",
                "Config1": "configs/lime/lime_delowlight_config1_ssim.yaml",
                "Config2": "configs/lime/lime_delowlight_config2_brisque.yaml",
                "Config3": None
            },
            "ds_fr": safe_init_dataset(LOLDataset, str(base_dir / "data/lol/splits/test.txt"), str(base_dir / "dataset/LOL")),
            "ds_nr": None
        },
        "BM3D": {
            "type": "bm3d", "img_fr": "noisy", "gt_fr": "clean",
            "configs": {
                "Raw": None,
                "Default": "configs/bm3d/bm3d_default_noise25.yaml",
                "Config1": "configs/bm3d/bm3d_config1_bsd25_ssim.yaml",
                "Config2": "configs/bm3d/bm3d_config2_bsd25_brisque.yaml",
                "Config3": None
            },
            "ds_fr": safe_init_dataset(BSDDenoiseDataset, root=str(base_dir / "data/bsd_denoise"), noise_level=25, split='test'),
            "ds_nr": None
        },
        "RBCP": {
            "type": "rbcp", "img_fr": None, "gt_fr": None,
            "configs": {
                "Raw": None,
                "Default": "configs/rbcp/desand_rbcp_default.yaml",
                "Config1": None,
                "Config2": "configs/rbcp/desand_optuna_config2_brisque.yaml",
                "Config3": "configs/rbcp/desand_optuna_config3_map50.yaml"
            },
            "ds_fr": None,
            "ds_nr": safe_init_dataset(DawnDataset, str(base_dir / "data/dawn/splits/sand_test_pairs.csv"), "")
        },
        "MotionDeblur": {
            "type": "motion_deblur", "img_fr": "blur", "gt_fr": "sharp",
            "configs": {
                "Raw": None,
                "Default": "configs/motion_deblur/richardson_lucy_default.yaml",
                "Config1": "configs/motion_deblur/optuna_config1_ssim.yaml",
                "Config2": "configs/motion_deblur/optuna_config2_brisque.yaml",
                "Config3": None
            },
            "ds_fr": safe_init_dataset(GoProDataset, str(base_dir / "data/gopro/gopro_pairs.csv"), split='test'),
            "ds_nr": None
        }
    }

    all_results = []
    
    for f_name, f_data in configs.items():
        # Check dataset availability before running
        ds_fr_available = f_data["ds_fr"] is not None
        if ds_fr_available:
            try:
                # Test basic check to see if datasets can be loaded
                if len(f_data["ds_fr"]) == 0:
                    ds_fr_available = False
            except Exception:
                ds_fr_available = False
                
        ds_nr_available = f_data["ds_nr"] is not None
        if ds_nr_available:
            try:
                if len(f_data["ds_nr"]) == 0:
                    ds_nr_available = False
            except Exception:
                ds_nr_available = False
                
        if not ds_fr_available and not ds_nr_available:
            print(f"Skipping {f_name} evaluation since datasets are empty or not prepared.")
            continue
            
        for c_name, c_path in f_data["configs"].items():
            if c_path is None and c_name != "Raw": 
                continue
            
            print(f"Evaluating {f_name} - {c_name}...")
            if c_name == "Raw":
                filt = None
            else:
                full_path = base_dir / c_path
                if not full_path.exists():
                    print(f"  Missing config: {full_path}")
                    continue
                filt = create_filter(f_data["type"], str(full_path))

            res = {"Filter": f_name, "Config": c_name}
            
            # Phase 1: Full-Reference Metrics Evaluation
            if ds_fr_available:
                print("  Phase 1: FR Metrics")
                fr_cache_dir = os.path.join(str(paper_figures_dir), f_name, c_name, "fr_images")
                cache_dataset(filt, f_data["ds_fr"], f_data["img_fr"], fr_cache_dir, quick_test)
                ssim, psnr = eval_fr(f_data["ds_fr"], f_data["gt_fr"], fr_cache_dir, quick_test)
                res["Phase1_SSIM"] = ssim
                res["Phase1_PSNR"] = psnr
                
            # Phase 2: No-Reference Metrics & Object Detection Evaluation
            if ds_nr_available:
                print("  Phase 2: NR Metrics")
                nr_cache_dir = os.path.join(str(paper_figures_dir), f_name, c_name, "images")
                cache_dataset(filt, f_data["ds_nr"], "image", nr_cache_dir, quick_test)
                brisque, niqe = eval_nr(f_data["ds_nr"], nr_cache_dir, quick_test)
                res["Phase2_BRISQUE"] = brisque
                res["Phase2_NIQE"] = niqe
                
                print("  Phase 2: Detection")
                yolo_dir = os.path.join(str(paper_figures_dir), f_name, c_name)
                y_met = eval_yolo(f_data["ds_nr"], yolo_dir, yolo_weights_path, dataset_name=f_name, quick_test=quick_test)
                res["Phase2_mAP50"] = y_met.get("map50", 0.0)
                res["Phase2_mAP50-95"] = y_met.get("map50_95", 0.0)
                res["Phase2_meanIoU"] = y_met.get("mean_iou", 0.0)
                
            all_results.append(res)
            
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(paper_tables_dir / "all_metrics_summary.csv", index=False)
        
        # Save to LaTeX
        with open(paper_tables_dir / "metrics_table.tex", "w") as f:
            f.write(df.to_latex(index=False, float_format="%.4f"))
            
        print(f"Saved evaluation outputs to {paper_tables_dir}")
    else:
        print("No evaluation was run since datasets were not found.")
    
    print("ALL EVALUATIONS DONE.")

if __name__ == "__main__":
    run_evaluation()
