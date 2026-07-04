# Computer Vision Restoration Project

Pipeline:

```text
degraded image -> restoration (AdaIR / UDPNet / HOGFormer / classic filters) -> YOLO detection -> evaluation
```

## Repository Structure

```text
Computer_vision/
|-- configs/                  YAML configs for classic filters and searches
|-- data/                     Dataset instructions only
|-- checkpoints/              Checkpoint instructions only
|-- dl_nets/                  Network code adapted from paper repositories
|   |-- AdaIR/
|   |-- UDPNet/
|   |   |-- models/
|   |   `-- depth_anything_v2/  UDPNet depth helper dependency
|   `-- HOGFormer/
|-- src/
|   |-- datasets/             Dataset wrappers and train/test loaders
|   |-- detection/            YOLO detection helpers
|   |-- metrics/              Image-quality and detection metrics
|   |-- restoration/          Classic restoration filters
|   |-- optimization/         Optuna objective functions
|   |-- experiments/          Experiment entrypoints grouped by method
|   |   |-- adair/
|   |   |-- udpnet/
|   |   |   `-- _internal/      Team-written UDPNet workflow internals
|   |   |-- hogformer/
|   |   `-- classic/
|   |-- training/             Main training entrypoints
|   `-- utils/                Small reusable helpers
|-- results/                  Selected report/slide artifacts only
|-- outputs/                  Raw run outputs, ignored by Git
|-- requirements.txt
`-- .gitignore
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Data And Checkpoints

Real datasets and weights are not committed. See:

```text
data/README.md
checkpoints/README.md
```

Pass local paths through CLI arguments or YAML config instead of hardcoding machine-specific paths.

## Train

AdaIR original-setting training:

```powershell
python src/training/train_adair_original.py `
  --adair-repo dl_nets/AdaIR `
  --output-dir outputs/adair/train `
  --data-file-dir data_dir `
  --denoise-dir data/Train/Denoise `
  --gopro-dir data/Train/Deblur `
  --enhance-dir data/Train/Enhance `
  --derain-dir data/Train/Derain `
  --dehaze-dir data/Train/Dehaze
```

HOGFormer training:

```powershell
python src/training/train_hogformer_original.py `
  --output_path outputs/hogformer/train `
  --ckpt_dir checkpoints/hogformer `
  --denoise_dir data/Train/Denoise `
  --gopro_dir data/Train/Deblur `
  --enhance_dir data/Train/Enhance `
  --derain_dir data/Train/Derain `
  --dehaze_dir data/Train/Dehaze
```

UDPNet training entrypoint:

```powershell
python src/training/train_udpnet_dehazing.py `
  --config src/training/udpnet_pipeline.yaml `
  --data-root data `
  --output-dir outputs/udpnet/train `
  --method-root dl_nets/UDPNet `
  --model FSNet_UDPNet
```

Note: the UDPNet training entrypoint is currently a minimal migration stub. It loads config and model, but the full OTS training loop still needs to be ported before real UDPNet training.

## Evaluate All Methods

Classical filters, including report-oriented YOLO evaluation:

```powershell
python src/experiments/classic/evaluate_all.py `
  --yolo-weights checkpoints/yolo/yolov8.pt `
  --output-dir outputs/classic/evaluate_all
```

AdaIR restoration plus YOLO before/after comparison:

```powershell
python src/experiments/adair/compare_yolo.py `
  --adair-repo dl_nets/AdaIR `
  --ckpt checkpoints/adair/adair5d.ckpt `
  --input-dirs data/RTTS/JPEGImages data/DAWN/images `
  --yolo-weights checkpoints/yolo/yolov8.pt `
  --output-dir outputs/adair/yolo_compare
```

HOGFormer test/evaluation:

```powershell
python src/experiments/hogformer/test.py `
  --ckpt-path checkpoints/hogformer/best.ckpt `
  --output_path outputs/hogformer/test

python src/experiments/hogformer/evaluate_foggy.py `
  --finetuned_ckpt checkpoints/hogformer/best.ckpt `
  --data_root data/Foggy_Cityscapes `
  --output_path outputs/hogformer/eval_foggy
```

UDPNet full workflow:

```powershell
python src/experiments/udpnet/scan_datasets.py --config src/training/udpnet_pipeline.yaml
python src/experiments/udpnet/normalize_gt.py --config src/training/udpnet_pipeline.yaml
python src/experiments/udpnet/generate_depthmaps.py `
  --config src/training/udpnet_pipeline.yaml `
  --depth-weights checkpoints/depth_anything/depth_anything_v2_vits.pth
python src/experiments/udpnet/evaluate_pipeline.py `
  --config src/training/udpnet_pipeline.yaml `
  --yolo-weights checkpoints/yolo/yolov8.pt `
  --output-dir outputs/udpnet/evaluate
```

## Method Notes

- `src/experiments/udpnet/_internal/` is team-written workflow code for scanning datasets, normalizing labels, generating depth maps, loading UDPNet, running YOLO, and writing metrics.
- `src/experiments/udpnet/_internal/` is not the original UDPNet repository structure.
- `dl_nets/UDPNet/depth_anything_v2/` is a helper dependency used to generate depth maps for UDPNet; it is not a separate restoration method.

## AdaIR Utilities

Restore one image:

```powershell
python src/experiments/adair/restore_single.py --input-image path/to/input.png --ckpt checkpoints/adair/adair5d.ckpt --output outputs/adair/restore_single/restored.png
```

Domain-shift analysis:

```powershell
python src/experiments/adair/analyze_sots_vs_dawn_fog.py --sots-dir path/to/sots --dawn-fog-dir path/to/dawn/fog --output-dir results/adair/domain_shift/sots_vs_dawn_fog
python src/experiments/adair/analyze_foggycity_vs_sots.py --foggycity-dir path/to/foggycity --sots-dir path/to/sots --output-dir results/adair/domain_shift/foggycity_vs_sots
python src/experiments/adair/analyze_dawn_weather_shift.py --dawn-dir path/to/dawn --output-dir results/adair/domain_shift/dawn_weather
```

## Classic Filter Utilities

Prepare data, optimize filter parameters, and generate report figures:

```powershell
python src/experiments/classic/prepare_data.py --help
python src/experiments/classic/optuna_optimize.py --method dcp --config config1_reside_ssim --n-trials 20 --output-dir outputs/classic/optuna
python src/experiments/classic/visualize_paper_samples.py --yolo-weights checkpoints/yolo/yolov8.pt
```

## Conventions

- `dl_nets/` contains paper network code and method-specific dependencies needed for import/load only.
- `src/training/` contains main training scripts.
- `src/experiments/<method>/` contains public experiment, demo, comparison, and analysis entrypoints.
- `src/experiments/<method>/_internal/` contains method-specific workflow internals, not public entrypoints.
- `src/utils/` contains small reusable helpers only.
- `outputs/` is ignored and used for raw outputs.
- `results/` contains selected report artifacts only.
- `data/` and `checkpoints/` contain instructions, not real datasets or weights.