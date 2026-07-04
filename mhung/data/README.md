# Data Directory Placeholder

This directory is used to store processed datasets, split configuration files (val/test splits), and Optuna trials result databases.

## Directory Structure (Expected)
Once you run the preparation script (`src/experiments/prepare_data.py`), this directory will be structured as follows:

```text
data/
├── bsd_denoise/
│   ├── val.txt
│   ├── test.txt
│   ├── noise15/
│   ├── noise25/
│   └── noise50/
├── reside6k/
│   └── splits/
│       ├── val.txt
│       └── test.txt
├── lol/
│   ├── train/
│   ├── val/
│   └── test/
├── rain100h/
│   └── splits/
│       ├── val.txt
│       └── test.txt
├── snow100k/
│   └── splits/
│       ├── val.txt
│       └── test.txt
├── dawn/
│   ├── images/
│   ├── labels/
│   └── splits/
│       ├── fog_val_pairs.csv
│       └── fog_test_pairs.csv
└── ...
```

## How to Populate
Run the unified data preparation script from the project root:
```bash
python src/experiments/prepare_data.py --dataset all
```
*Note: Ensure that you have downloaded the raw datasets in the `dataset/` folder beforehand.*
