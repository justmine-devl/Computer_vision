# Raw Datasets Directory

This directory should contain the raw, unmodified datasets downloaded from their official sources. They will be processed and split using the preparation script.

## Expected Directory Structure
Please arrange your downloaded raw datasets as follows:

```text
dataset/
├── BSD/                      # BSD (Berkeley Segmentation Dataset) Denoise dataset
│   ├── BSD_noisy15/
│   ├── BSD_noisy25/
│   └── BSD_noisy50/
│
├── RESIDE-6K/                # RESIDE-6K Dehazing dataset
│   └── test/
│       ├── hazy/
│       └── GT/
│
├── DAWN/                     # DAWN Abnormal Weather dataset
│   ├── Fog/
│   ├── Rain/
│   ├── Sand/
│   └── Snow/
│
├── LOL/                      # LOL Low-Light enhancement dataset
│   ├── our485/
│   └── eval15/
│
├── rain100H/                 # rain100H Deraining dataset
│   └── train/
│       ├── rain/
│       └── norain/
│
├── GoPro/                    # GoPro Deblurring dataset
│   ├── input/
│   └── sharp/
│
├── Snow100K/                 # Snow100K Desnowing dataset
│   ├── synthetic/
│   └── gt/
│
└── RainDrop/                 # RainDrop Deraining dataset
    ├── input/
    └── target/
```

## How to Prepare
Once the raw datasets are correctly structured here, run:
```bash
python src/experiments/prepare_data.py --dataset all
```
This will automatically parse the datasets, create val/test splits, and store the configuration splits in the `data/` directory.

*Note: Raw images and large dataset folders are ignored by Git and will not be pushed to GitHub.*
