# Data Directory

This folder is intended to host training and testing datasets for Computer Vision models.

## Guideline

* **Do NOT commit or push real datasets** to the Git repository.
* Large datasets should reside locally and be ignored by Git (using the `.gitignore` rules).

## Foggy Cityscapes Dataset Layout

To train or evaluate on the Foggy Cityscapes dataset, please construct a directory named `foggy_cityscapes` here with the following layout:

```text
foggy_cityscapes/
├── train/
│   ├── input/     # Contains hazy images (e.g., _fog_beta_0.02.png)
│   └── target/    # Contains matching clean ground truth images
└── val/
    ├── input/
    └── target/
```
