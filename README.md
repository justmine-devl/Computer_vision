# Improving Object Detection in Unconstrained Environments via Image Restoration

This repository contains the complete source code for our Computer Vision project. The goal is to improve object detection under adverse weather/degradation conditions by restoring degraded images before YOLO detection.

We implement and compare **four categories of restoration approaches**:

| Category | Methods | Type |
|----------|---------|------|
| **AdaIR** | Adaptive Image Restoration (5-degradation) | Deep Learning |
| **UDPNet** | Uncertainty-aware Dehazing (ConvIR / FSNet) | Deep Learning |
| **HOGformer** | HOG-based Transformer for Image Restoration | Deep Learning |
| **Classic Filters** | DCP, WMGF, LIME, BM3D, RBCP, Desnowing, Motion Deblur (Wiener / Richardson-Lucy / Fergus) | Prior-based |

Pipeline:

```text
degraded image -> restoration (AdaIR / UDPNet / HOGformer / Classic Filters) -> YOLO detection -> evaluation
```

---

## Repository Structure

```text
Computer_vision/
|-- configs/                  YAML parameter configs for classic filters
|-- data/                     Data instructions only. Real datasets are not committed.
|-- checkpoints/              Checkpoint instructions only. Real weights are not committed.
|-- dl_nets/                  Network architectures adapted from paper repos
|   |-- AdaIR/
|   |-- UDPNet/
|   `-- HOGformer/
|-- src/
|   |-- datasets/             PyTorch dataset wrappers and restoration train/test loaders
|   |-- detection/            YOLO detection runner and helpers
|   |-- metrics/              Image-quality and detection metrics
|   |-- restoration/          Classic prior-based restoration filters
|   |-- optimization/         Optuna objective functions
|   |-- experiments/          Experiment and analysis entrypoints
|   |-- training/             Training entrypoints and configs
|   |-- pipelines/            End-to-end model pipelines
|   |   `-- udpnet_pipeline/  UDPNet-specific pipeline utilities
|   `-- utils/                Shared image, plotting, scheduler, and metric helpers
|-- results/                  Selected figures/CSV files for report/slides
|-- requirements.txt
`-- .gitignore
```

---

## Getting Started

### Installation

```bash
pip install -r requirements.txt
```

### Data Setup

Download datasets and place them locally (see `data/README.md` for structure).
All paths are passed through command-line arguments â€” no hardcoded paths.

### Checkpoint Setup

Place model weights in `checkpoints/` (see `checkpoints/README.md` for layout).

---

## Methods

### 1. AdaIR â€” Adaptive Image Restoration

AdaIR handles multiple degradation types (haze, rain, noise, snow, low-light) with a single model.

**Train:**
```bash
python src/training/train_adair_original.py --adair-repo dl_nets/AdaIR --output-dir outputs/adair
```

**Restore a single image:**
```bash
python src/experiments/adair_restore_single.py --input-image path/to/input.png --ckpt checkpoints/adair/adair5d.ckpt --output results/adair/restored.png
```

**Compare YOLO detection (before/after):**
```bash
python src/experiments/adair_compare_yolo.py --ckpt checkpoints/adair/adair5d.ckpt --data-dir path/to/data
```

**Domain shift analysis:**
```bash
python src/experiments/adair_analyze_sots_vs_dawn_fog.py --sots-dir path/to/sots --dawn-fog-dir path/to/dawn/fog --output-dir results/adair/domain_shift/
python src/experiments/adair_analyze_foggycity_vs_sots.py --foggycity-dir path/to/foggycity --sots-dir path/to/sots --output-dir results/adair/domain_shift/
python src/experiments/adair_analyze_dawn_weather_shift.py --dawn-dir path/to/dawn --output-dir results/adair/domain_shift/
```

### 2. UDPNet â€” Uncertainty-aware Dehazing

UDPNet specializes in image dehazing using depth-aware uncertainty estimation.

**Train:**
```bash
python src/training/train_udpnet_dehazing.py --config src/training/udpnet_pipeline.yaml
```

**Evaluate pipeline:**
```bash
python src/experiments/udpnet_evaluate_pipeline.py --config src/training/udpnet_pipeline.yaml
```

**Other utilities:**
```bash
python src/experiments/udpnet_extract_foggycityscape.py --data-dir path/to/cityscapes
python src/experiments/udpnet_generate_depthmaps.py --input-dir path/to/images
python src/experiments/udpnet_compare_panels.py --results-dir path/to/results
python src/experiments/udpnet_scan_datasets.py --data-dir path/to/datasets
python src/experiments/udpnet_normalize_gt.py --input-dir path/to/gt
```

### 3. HOGformer â€” HOG-based Transformer

HOGformer leverages Histogram of Oriented Gradients within a transformer architecture for image restoration.

**Train (standard):**
```bash
python src/training/train_hogformer_lightning.py
```

**Train on foggy data:**
```bash
python src/training/train_hogformer_foggy.py --data-dir path/to/foggy/data
```

**Train with learning curves:**
```bash
python src/training/train_hogformer_curve.py
```

**Test:**
```bash
python src/experiments/test_hogformer.py --ckpt checkpoints/hogformer/best.ckpt
```

**Evaluate on foggy data:**
```bash
python src/experiments/evaluate_hogformer_foggy.py --ckpt checkpoints/hogformer/best.ckpt --data-dir path/to/foggy/data
```

### 4. Classic Prior-based Filters

Classic image restoration methods with Optuna hyperparameter optimization.

**Available methods:** DCP (dehazing), WMGF (deraining), LIME (low-light), BM3D (denoising), RBCP (desanding), Morphological Desnowing, Motion Deblur (Wiener / Richardson-Lucy / Fergus blind).

**Data preparation:**
```bash
python src/experiments/prepare_data.py --dataset all
```

**Hyperparameter search with Optuna:**
```bash
python src/experiments/optuna_optimize.py --method <method_name> --config <config_name> --n-trials 20
```
- `method_name` choices: `dcp`, `wmgf`, `desnow`, `lime`, `bm3d`, `rbcp`, `motion`
- `config_name` examples: `config1_reside_ssim`, `config2_dawn_brisque`, `config3_dawn_map50`

**Full evaluation:**
```bash
python src/experiments/evaluate_all.py --yolo-weights checkpoints/yolo26n.pt
```

**Visual comparison grids:**
```bash
python src/experiments/visualize_paper_samples.py --yolo-weights checkpoints/yolo26n.pt
```

---

## Conventions

- `dl_nets/` contains network code adapted from paper repositories, with unnecessary docs/data/checkpoints removed.
- `src/training/` contains main training files.
- `src/experiments/` contains experiment, comparison, and analysis entrypoints.
- `src/datasets/` contains dataset loaders; `src/utils/` contains reusable helper code only.
- `results/` contains selected report artifacts only, not raw outputs.
- `data/` and `checkpoints/` must not contain real files in Git.
- All paths must be passed through parser arguments. Do not hardcode local paths.
- Run scripts from the repository root so their path setup can find `src/` and `dl_nets/`.

