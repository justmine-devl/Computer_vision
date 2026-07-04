# Improving Object Detection in Unconstrained Environments via Image Restoration

This repository contains the AdaIR code used in the project. The goal is to improve object detection under adverse weather by restoring degraded images before YOLO detection.

Pipeline:

```text
degraded image -> AdaIR restoration -> YOLO detection -> evaluation
```

## Structure

```text
data/                 Data instructions only. Real datasets are not committed.
checkpoints/          Checkpoint instructions only. Real weights are not committed.
dl_nets/AdaIR/        Minimal AdaIR network code needed for import/load.
src/training/         Main training scripts.
src/experiments/      Experiment, comparison, and analysis scripts.
src/utils/            Reusable utilities shared by training/experiments.
results/adair/        Selected small figures/CSV files for report/slides.
```

## Conventions

- `dl_nets/` contains network code adapted from paper repositories, with unnecessary docs/data/checkpoints removed.
- `src/training/` contains main training files.
- `src/experiments/` contains one-off experiment, comparison, and analysis entrypoints.
- `src/utils/` contains code reused by at least two scripts.
- `results/` contains selected report artifacts only, not raw outputs.
- `data/` and `checkpoints/` must not contain real files in Git.
- All paths must be passed through parser arguments. Do not hardcode local paths.

## Examples

Restore a single image:

```bash
python src/experiments/adair_restore_single.py --input-image path/to/input.png --ckpt checkpoints/adair/adair5d.ckpt --output results/adair/restoration_yolo/restored.png
```

Analyze SOTS vs DAWN-Fog domain shift:

```bash
python src/experiments/adair_analyze_sots_vs_dawn_fog.py --sots-dir path/to/sots --dawn-fog-dir path/to/dawn/fog --output-dir results/adair/domain_shift/sots_vs_dawn_fog
```
