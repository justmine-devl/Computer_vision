import os
from ultralytics import YOLO
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.metrics.detection import extract_detection_metrics

class YOLOEvaluator:
    def __init__(self, model_path: str, data_yaml: str, device: str = "auto"):
        self.model = YOLO(model_path)
        self.data_yaml = data_yaml
        self.device = None if device == "auto" else device

    def evaluate(self, image_dir_or_yaml: str, classes=None) -> dict:
        results = self.model.val(data=self.data_yaml, split='val', device=self.device, verbose=False, classes=classes)
        metrics = extract_detection_metrics(results)
        return metrics

    def predict_and_save(self, image_path: str, output_dir: str, classes=None):
        """Predict and save the image with bounding boxes to output_dir"""
        os.makedirs(output_dir, exist_ok=True)
        # model.predict returns list of Results
        res = self.model.predict(source=image_path, device=self.device, save=True, project=output_dir, name="predict", exist_ok=True, classes=classes)
        return res
