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

import numpy as np
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