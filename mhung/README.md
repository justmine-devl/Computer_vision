# Image Restoration & Object Detection (Abnormal Weather Module)

This folder contains the code for the image restoration and object detection experiments under abnormal weather/degradation conditions (Haze, Rain, Snow, Low-light, Noise, Sand-dust, and Motion blur). 

It is a sub-module of our main Computer Vision project, focusing on implementing classic prior-based restoration algorithms and evaluating their impact on YOLOv8 object detection performance.

---

## What's in this folder?

Here is a quick look at the directory structure:

*   **`src/`**: All the source code files.
    *   `restoration/`: Contains the actual filter classes (DCP, WMGF, LIME, RBCP, BM3D, Richardson-Lucy, etc.).
    *   `datasets/`: PyTorch-like dataset wrappers for loading images (RESIDE, GoPro, LOL, DAWN, BSDDenoise).
    *   `detection/`: Helper classes for running YOLOv8 validation and computing mean IoU.
    *   `metrics/`: Code for measuring full-reference (PSNR, SSIM) and no-reference (BRISQUE, NIQE) image quality.
    *   `optimization/`: Optuna objective functions used during hyperparameter search.
    *   `experiments/`: Executable scripts to run the pipeline (data prep, optimization, evaluation, visualization).
*   **`configs/`**: Storage for YAML parameter configurations. It has folders for each restoration method containing default parameters and Optuna-optimized parameters (optimized for SSIM, BRISQUE, or YOLO $mAP_{50}$).
*   **`checkpoints/`**: Directory where you should place the YOLO weights (e.g., `yolo26n.pt`).
*   **`data/`**: Text-based metadata splits (Val/Test pairs). The raw images are not kept here.
*   **`dataset/`**: The folder where raw downloaded datasets should be placed (ignored by Git).
*   **`paper/`**: Local folder containing generated outputs for our report:
    *   `tables/`: Output tables in `.csv` and LaTeX format.
    *   `figures/`: Side-by-side comparison grids showing restoration and YOLO detection boxes.

---

## How to run this module

First, install the required packages:
```bash
pip install -r requirements.txt
```

### 1. Data Setup & Split Generation
Download the datasets and put them in the `dataset/` folder (check the format inside `dataset/README.md`). Then run the preparation script to generate splits:
```bash
python src/experiments/prepare_data.py --dataset all
```

### 2. Search for Optimal Parameters (Optuna)
To run Optuna and find the best hyper-parameters for a filter based on SSIM, BRISQUE, or YOLO $mAP_{50}$:
```bash
python src/experiments/optuna_optimize.py --method <method_name> --config <config_name> --n-trials 20
```
*   `method_name` choices: `dcp`, `wmgf`, `desnow`, `lime`, `bm3d`, `rbcp`, `motion`.
*   `config_name` examples: `config1_reside_ssim`, `config2_dawn_brisque`, `config3_dawn_map50` (or `map50` for other methods).

### 3. Evaluate Everything
To run the full evaluation across all weather conditions and configurations (computes restoration metrics, runs YOLO validation, and writes LaTeX tables directly into `paper/tables/`):
```bash
python src/experiments/evaluate_all.py --yolo-weights checkpoints/yolo26n.pt
```
*If you just want to test if the pipeline runs without errors, use:*
```bash
python src/experiments/evaluate_all.py --quick-test
```

### 4. Create Visual Grids for the Report
To generate the comparison grids with YOLO detection bounding box overlays:
```bash
python src/experiments/visualize_paper_samples.py --yolo-weights checkpoints/yolo26n.pt
```
*For a quick dry-run test:*
```bash
python src/experiments/visualize_paper_samples.py --quick-test
```
*   The results will be saved under the `paper/figures/` folder.
