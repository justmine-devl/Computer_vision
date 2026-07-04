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
|   `-- HOGFormer/
|-- src/
|   |-- datasets/             Dataset wrappers and train/test loaders
|   |-- detection/            YOLO detection helpers
|   |-- metrics/              Image-quality and detection metrics
|   |-- restoration/          Classic restoration filters
|   |-- optimization/         Optuna objective functions
|   |-- pipelines/            Multi-module method pipelines
|   |   `-- udpnet/
|   |-- experiments/          Experiment entrypoints grouped by method
|   |   |-- adair/
|   |   |-- udpnet/
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

Pass local paths through CLI arguments instead of hardcoding machine-specific paths.

## AdaIR

Train:

```powershell
python src/training/train_adair_original.py --adair-repo dl_nets/AdaIR --output-dir outputs/adair/train
```

Restore one image:

```powershell
python src/experiments/adair/restore_single.py --input-image path/to/input.png --ckpt checkpoints/adair/adair5d.ckpt --output outputs/adair/restore_single/restored.png
```

Compare YOLO before and after restoration:

```powershell
python src/experiments/adair/compare_yolo.py --ckpt checkpoints/adair/adair5d.ckpt --input-dirs path/to/images --output-dir outputs/adair/yolo_compare
```

Domain-shift analysis:

```powershell
python src/experiments/adair/analyze_sots_vs_dawn_fog.py --sots-dir path/to/sots --dawn-fog-dir path/to/dawn/fog --output-dir results/adair/domain_shift/sots_vs_dawn_fog
python src/experiments/adair/analyze_foggycity_vs_sots.py --foggycity-dir path/to/foggycity --sots-dir path/to/sots --output-dir results/adair/domain_shift/foggycity_vs_sots
python src/experiments/adair/analyze_dawn_weather_shift.py --dawn-dir path/to/dawn --output-dir results/adair/domain_shift/dawn_weather
```

## UDPNet

Train entrypoint:

```powershell
python src/training/train_udpnet_dehazing.py --config src/training/udpnet_pipeline.yaml
```

Pipeline experiments:

```powershell
python src/experiments/udpnet/evaluate_pipeline.py --config src/training/udpnet_pipeline.yaml
python src/experiments/udpnet/scan_datasets.py --config src/training/udpnet_pipeline.yaml
python src/experiments/udpnet/normalize_gt.py --config src/training/udpnet_pipeline.yaml
python src/experiments/udpnet/generate_depthmaps.py --config src/training/udpnet_pipeline.yaml
python src/experiments/udpnet/compare_panels.py --help
```

## HOGFormer

Train/evaluate:

```powershell
python src/training/train_hogformer_original.py --help
python src/experiments/hogformer/test.py --ckpt-path checkpoints/hogformer/best.ckpt --output_path outputs/hogformer/test/
python src/experiments/hogformer/evaluate_foggy.py --finetuned_ckpt checkpoints/hogformer/best.ckpt --data_root path/to/foggy/data --output_path outputs/hogformer/eval_foggy/
```

## Classic Filters

Prepare, optimize, evaluate, and generate report figures:

```powershell
python src/experiments/classic/prepare_data.py --help
python src/experiments/classic/optuna_optimize.py --method dcp --config config1_reside_ssim --n-trials 20 --output-dir outputs/classic/optuna
python src/experiments/classic/evaluate_all.py --yolo-weights checkpoints/yolo/yolo26n.pt --output-dir outputs/classic/evaluate_all
python src/experiments/classic/visualize_paper_samples.py --yolo-weights checkpoints/yolo/yolo26n.pt
```

## Conventions

- `dl_nets/` contains paper network code needed for import/load only.
- `src/training/` contains main training scripts.
- `src/experiments/<method>/` contains experiment, demo, comparison, and analysis entrypoints.
- `src/pipelines/` contains larger multi-module pipelines.
- `src/utils/` contains small reusable helpers only.
- `outputs/` is ignored and used for raw outputs.
- `results/` contains selected report artifacts only.
- `data/` and `checkpoints/` contain instructions, not real datasets or weights.
