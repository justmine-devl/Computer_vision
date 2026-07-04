# Source Layout

`src/datasets/` contains PyTorch-like dataset wrappers for loading images (RESIDE, GoPro, LOL, DAWN, BSD, Snow100K, etc.).

`src/detection/` contains YOLO detection runner and helpers.

`src/metrics/` contains image quality metrics: full-reference (PSNR, SSIM), no-reference (BRISQUE, NIQE), and detection (mAP).

`src/restoration/` contains classic prior-based restoration filter implementations (DCP, WMGF, LIME, BM3D, RBCP, Desnowing, Motion Deblur).

`src/optimization/` contains Optuna objective functions for hyperparameter search across restoration methods.

`src/training/` contains main training files for all methods (AdaIR, UDPNet, HOGFormer).

`src/experiments/<method>/` contains experiment, comparison, and analysis scripts. If code is only used once, keep it here.

`src/utils/` contains reusable utilities. If code is reused by two or more scripts, move it here.

Run scripts from the repository root so their path setup can find `src/` and `dl_nets/`.
