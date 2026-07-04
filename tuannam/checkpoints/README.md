# Checkpoints Directory

This directory is intended to host trained model checkpoints (weights) for inference and fine-tuning.

## Guideline

* **Do NOT commit or push model weights** (`.ckpt`, `.pth`, `.pt`, etc.) to the Git repository.
* Model weights should be stored locally and are ignored by Git.

## Expected Pretrained Models

Place the following pretrained model weights inside this folder:

* `adair5d.ckpt` - Pretrained model on 5-degradations.
* `latest.pth` / `best_psnr.pth` - Local fine-tuned weights for validation and evaluation.
