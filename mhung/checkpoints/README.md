# Checkpoints Directory Placeholder

This directory is designated for storing pre-trained model weights and checkpoints (such as YOLO detection model weights).

## Expected Files
You should place the pre-trained detection model weights in this directory:
- `yolo8n.pt` (or any other YOLO weights utilized in evaluations)

## How to Get YOLO Weights
If you do not have the weights, you can download them using Python:
```python
from ultralytics import YOLO
# This will automatically download the model weights to the current directory
model = YOLO("yolo8n.pt") 
```
After downloading, move the `.pt` file to this `checkpoints/` directory.

The evaluation scripts will look for YOLO weights here by default, e.g.:
```bash
python src/experiments/evaluate_all.py --yolo-weights checkpoints/yolo8n.pt
```
*Note: Large weights files (`*.pt`, `*.pth`) are ignored by Git in this repository.*
