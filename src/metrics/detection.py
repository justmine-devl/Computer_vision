from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

def extract_detection_metrics(val_results) -> dict:
    """
    Extract relevant metrics from ultralytics validation results object.
    """
    metrics = val_results.results_dict
    
    def get_key(k_list):
        for k in k_list:
            if k in metrics:
                return metrics[k]
        return 0.0

    p = get_key(['metrics/precision(B)', 'precision'])
    r = get_key(['metrics/recall(B)', 'recall'])
    map50 = get_key(['metrics/mAP50(B)', 'mAP50'])
    map50_95 = get_key(['metrics/mAP50-95(B)', 'mAP50-95'])
    
    # F1-score
    eps = 1e-7
    f1 = 2 * p * r / (p + r + eps)
    
    res = {
        "map50": map50,
        "map50_95": map50_95,
        "precision": p,
        "recall": r,
        "f1": f1
    }
    
    try:
        if hasattr(val_results, 'box') and val_results.box is not None:
            ap50_per_class = val_results.box.ap50
            class_ids = val_results.box.ap_class_index
            names = val_results.names
            
            for i, c_id in enumerate(class_ids):
                name = names.get(c_id, f"class_{c_id}")
                res[f"ap_{name}"] = ap50_per_class[i]
    except Exception as e:
        print(f"Warning: Could not extract per-class AP: {e}")
        
    return res

# ===================================== #

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0

def compute_mean_iou(yolo_model, images_dir, labels_dir, allowed_classes=[0, 1, 2, 3, 5, 6, 7]):
    import glob, os, cv2
    image_files = glob.glob(os.path.join(images_dir, '*.jpg')) + glob.glob(os.path.join(images_dir, '*.png'))
    ious = []
    
    for img_path in image_files:
        basename = os.path.basename(img_path)
        label_path = os.path.join(labels_dir, os.path.splitext(basename)[0] + '.txt')
        
        if not os.path.exists(label_path): continue
            
        gt_boxes = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    if cls_id not in allowed_classes: continue
                    if len(parts) == 5:
                        gt_boxes.append({
                            'cls': cls_id, 'x': float(parts[1]), 'y': float(parts[2]),
                            'w': float(parts[3]), 'h': float(parts[4])
                        })
                    else:
                        coords = list(map(float, parts[1:]))
                        xs = coords[0::2]
                        ys = coords[1::2]
                        x_min, x_max = min(xs), max(xs)
                        y_min, y_max = min(ys), max(ys)
                        gt_boxes.append({
                            'cls': cls_id, 'x': (x_min + x_max) / 2, 'y': (y_min + y_max) / 2,
                            'w': x_max - x_min, 'h': y_max - y_min
                        })
                    
        if len(gt_boxes) == 0: continue
            
        img = cv2.imread(img_path)
        if img is None: continue
        h, w = img.shape[:2]
        
        for gt in gt_boxes:
            x_center, y_center = gt['x'] * w, gt['y'] * h
            box_w, box_h = gt['w'] * w, gt['h'] * h
            gt['box'] = [x_center - box_w/2, y_center - box_h/2, x_center + box_w/2, y_center + box_h/2]
            
        res = yolo_model.predict(img, classes=allowed_classes, verbose=False)[0]
        preds = []
        if res.boxes is not None:
            for i in range(len(res.boxes)):
                cls_id = int(res.boxes.cls[i].item())
                box = res.boxes.xyxy[i].cpu().numpy()
                preds.append({'cls': cls_id, 'box': box})
                
        for gt in gt_boxes:
            best_iou = 0
            for p in preds:
                if p['cls'] == gt['cls']:
                    iou = calculate_iou(gt['box'], p['box'])
                    if iou > best_iou: best_iou = iou
            
            # Only count if the object was actually detected (True Positive)
            if best_iou > 0:
                ious.append(best_iou)
            
    if len(ious) == 0: return 0.0
    return np.mean(ious)

def box_iou_xyxy(box_a: np.ndarray, box_b: np.ndarray) -> float:
        x1 = max(float(box_a[0]), float(box_b[0]))
        y1 = max(float(box_a[1]), float(box_b[1]))
        x2 = min(float(box_a[2]), float(box_b[2]))
        y2 = min(float(box_a[3]), float(box_b[3]))

        inter_w = max(0.0, x2 - x1)
        inter_h = max(0.0, y2 - y1)
        inter_area = inter_w * inter_h

        area_a = max(0.0, float(box_a[2]) - float(box_a[0])) * max(
                0.0, float(box_a[3]) - float(box_a[1])
        )
        area_b = max(0.0, float(box_b[2]) - float(box_b[0])) * max(
                0.0, float(box_b[3]) - float(box_b[1])
        )
        denom = area_a + area_b - inter_area + 1e-9
        return inter_area / denom


def extract_predictions_from_ultralytics(result_obj: Any) -> np.ndarray:
        boxes = getattr(result_obj, "boxes", None)
        if boxes is None or boxes.xyxy is None:
                return np.zeros((0, 6), dtype=np.float32)

        xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        conf = boxes.conf.detach().cpu().numpy().astype(np.float32)
        cls = boxes.cls.detach().cpu().numpy().astype(np.float32)

        out = np.zeros((xyxy.shape[0], 6), dtype=np.float32)
        out[:, 0] = cls
        out[:, 1:5] = xyxy
        out[:, 5] = conf
        return out


def mean_iou_all_gt(preds: Sequence[np.ndarray], gts: Sequence[np.ndarray]) -> float:
        scores: List[float] = []
        for pred_arr, gt_arr in zip(preds, gts):
                if gt_arr.shape[0] == 0:
                        continue

                if pred_arr.shape[0] == 0:
                        scores.extend([0.0] * int(gt_arr.shape[0]))
                        continue

                for gt in gt_arr:
                        gt_cls = int(gt[0])
                        gt_box = gt[1:5]
                        matched = pred_arr[
                                pred_arr[:, 0].astype(np.int32) == gt_cls
                        ][:, 1:5]
                        best = 0.0
                        for box in matched:
                                best = max(best, box_iou_xyxy(gt_box, box))
                        scores.append(best)

        return float(np.mean(scores)) if scores else 0.0


def mean_iou_matched(preds: Sequence[np.ndarray], gts: Sequence[np.ndarray]) -> float:
        scores: List[float] = []
        for pred_arr, gt_arr in zip(preds, gts):
                if gt_arr.shape[0] == 0 or pred_arr.shape[0] == 0:
                        continue

                for gt in gt_arr:
                        gt_cls = int(gt[0])
                        gt_box = gt[1:5]
                        matched = pred_arr[
                                pred_arr[:, 0].astype(np.int32) == gt_cls
                        ][:, 1:5]
                        if matched.shape[0] == 0:
                                continue
                        best = 0.0
                        for box in matched:
                                best = max(best, box_iou_xyxy(gt_box, box))
                        if best > 0.0:
                                scores.append(best)

        return float(np.mean(scores)) if scores else 0.0


def mean_best_iou(preds: Sequence[np.ndarray], gts: Sequence[np.ndarray]) -> float:
        return mean_iou_matched(preds, gts)


def mean_iou_tp50(preds: Sequence[np.ndarray], gts: Sequence[np.ndarray]) -> float:
        scores: List[float] = []

        for pred_arr, gt_arr in zip(preds, gts):
                if gt_arr.shape[0] == 0 or pred_arr.shape[0] == 0:
                        continue

                class_ids = sorted(
                        set(gt_arr[:, 0].astype(np.int32).tolist())
                        | set(pred_arr[:, 0].astype(np.int32).tolist())
                )

                for cls_id in class_ids:
                        gt_cls = gt_arr[gt_arr[:, 0].astype(np.int32) == cls_id]
                        pred_cls = pred_arr[pred_arr[:, 0].astype(np.int32) == cls_id]
                        if gt_cls.shape[0] == 0 or pred_cls.shape[0] == 0:
                                continue

                        order = np.argsort(-pred_cls[:, 5])
                        matched_gt = np.zeros(gt_cls.shape[0], dtype=np.bool_)

                        for pred_idx in order:
                                pred_box = pred_cls[pred_idx, 1:5]
                                ious = np.array(
                                        [
                                                box_iou_xyxy(pred_box, gt_row[1:5])
                                                for gt_row in gt_cls
                                        ],
                                        dtype=np.float32,
                                )
                                best_idx = int(np.argmax(ious))
                                best_iou = float(ious[best_idx])

                                if best_iou >= 0.50 and not matched_gt[best_idx]:
                                        matched_gt[best_idx] = True
                                        scores.append(best_iou)

        return float(np.mean(scores)) if scores else 0.0


def _compute_ap_for_class(
        cls_id: int,
        preds_per_image: Sequence[np.ndarray],
        gts_per_image: Sequence[np.ndarray],
        iou_thr: float,
) -> float:
        total_gt = 0
        detections: List[Tuple[int, float, np.ndarray]] = []
        gt_by_image: Dict[int, np.ndarray] = {}

        for img_idx, gt_arr in enumerate(gts_per_image):
                gt_cls = gt_arr[gt_arr[:, 0].astype(np.int32) == cls_id]
                gt_by_image[img_idx] = gt_cls
                total_gt += gt_cls.shape[0]

                pred_arr = preds_per_image[img_idx]
                pred_cls = pred_arr[pred_arr[:, 0].astype(np.int32) == cls_id]
                for row in pred_cls:
                        detections.append((img_idx, float(row[5]), row[1:5]))

        if total_gt == 0:
                return np.nan
        if not detections:
                return 0.0

        detections.sort(key=lambda x: x[1], reverse=True)

        matched_flags = {
                img_idx: np.zeros(gt_by_image[img_idx].shape[0], dtype=np.bool_)
                for img_idx in range(len(gts_per_image))
        }

        tp = np.zeros(len(detections), dtype=np.float32)
        fp = np.zeros(len(detections), dtype=np.float32)

        for det_idx, (img_idx, _conf, pred_box) in enumerate(detections):
                gt_arr = gt_by_image[img_idx]
                if gt_arr.shape[0] == 0:
                        fp[det_idx] = 1.0
                        continue

                ious = np.array(
                        [box_iou_xyxy(pred_box, gt[1:5]) for gt in gt_arr],
                        dtype=np.float32,
                )
                best_idx = int(np.argmax(ious))
                best_iou = float(ious[best_idx])

                if best_iou >= iou_thr and not matched_flags[img_idx][best_idx]:
                        tp[det_idx] = 1.0
                        matched_flags[img_idx][best_idx] = True
                else:
                        fp[det_idx] = 1.0

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        if float(tp_cum[-1]) == 0.0:
                return 0.0
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
        recall = tp_cum / max(total_gt, 1)

        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0]))

        for idx in range(mpre.size - 1, 0, -1):
                mpre[idx - 1] = max(mpre[idx - 1], mpre[idx])

        points = np.linspace(0.0, 1.0, 101)
        ap = np.mean(
                [np.max(mpre[mrec >= p]) if np.any(mrec >= p) else 0.0 for p in points]
        )
        return float(ap)


def compute_map(
        preds_per_image: Sequence[np.ndarray],
        gts_per_image: Sequence[np.ndarray],
        iou_thresholds: Sequence[float],
        class_set_mode: str = "gt_only",
) -> Dict[str, Any]:
        class_set_mode = str(class_set_mode).lower()
        gt_classes = {int(v) for arr in gts_per_image for v in arr[:, 0].tolist()}
        pred_classes = {
                int(v) for arr in preds_per_image for v in arr[:, 0].tolist()
        }
        if class_set_mode == "gt_union_pred":
                classes = sorted(gt_classes | pred_classes)
        else:
                classes = sorted(gt_classes)

        if not classes:
                return {
                        "classes": [],
                        "map_by_iou": {str(round(t, 2)): 0.0 for t in iou_thresholds},
                        "map": 0.0,
                        "ap_per_class": {},
                        "class_set_mode": class_set_mode,
                }

        map_by_iou: Dict[str, float] = {}

        # Also collect AP per class at the primary IoU threshold (first threshold)
        ap_per_class: Dict[int, float] = {}
        for thr in iou_thresholds:
                ap_values: List[float] = []
                for cls_id in classes:
                        ap = _compute_ap_for_class(
                                cls_id, preds_per_image, gts_per_image, float(thr)
                        )
                        if np.isnan(ap) and class_set_mode == "gt_union_pred":
                                ap = 0.0
                        if not np.isnan(ap):
                                ap_values.append(ap)
                        # record AP per class for the primary threshold
                        if thr == iou_thresholds[0]:
                                ap_per_class[int(cls_id)] = (
                                        float(ap) if not np.isnan(ap) else 0.0
                                )
                map_by_iou[str(round(float(thr), 2))] = (
                        float(np.mean(ap_values)) if ap_values else 0.0
                )

        overall = float(np.mean(list(map_by_iou.values()))) if map_by_iou else 0.0
        return {
                "classes": classes,
                "map_by_iou": map_by_iou,
                "map": overall,
                "ap_per_class": ap_per_class,
                "class_set_mode": class_set_mode,
        }
