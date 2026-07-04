import os
import glob
import random
import shutil
import csv
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

# Standard COCO classes to YOLO class IDs mapping
COCO_CLASS_TO_ID = {
    "person": 0,
    "bicycle": 1,
    "car": 2,
    "motorcycle": 3,
    "bus": 5,
    "train": 6,
    "truck": 7
}

def get_project_root():
    # Detect project root relative to this script (three parent directories up from src/experiments/<group>)
    return Path(__file__).resolve().parents[3]

def convert_to_yolo(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def get_subset(pairs, seed=42):
    random.seed(seed)
    total = len(pairs)
    if total > 10000:
        subset_size = max(int(0.1 * total), 1000)
        return random.sample(pairs, subset_size)
    return pairs

def split_and_save_pairs(pairs, split_dir, seed=42):
    random.seed(seed)
    random.shuffle(pairs)
    split_idx = int(0.8 * len(pairs))
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]
    
    os.makedirs(split_dir, exist_ok=True)
    
    with open(os.path.join(split_dir, "train_pairs.csv"), "w") as f:
        for img, lbl in train_pairs:
            f.write(f"{img},{lbl}\n")
            
    with open(os.path.join(split_dir, "val_test_pairs.csv"), "w") as f:
        for img, lbl in val_pairs:
            f.write(f"{img},{lbl}\n")
            
    print(f"Saved to {split_dir} - Train: {len(train_pairs)}, Val/Test: {len(val_pairs)}")

# -------------------------------------------------------------
# Dataset Preparation Functions
# -------------------------------------------------------------

def prepare_bsd(project_root):
    print("Preparing BSD Denoise dataset...")
    random.seed(42)
    
    src_dir = Path(project_root) / 'dataset' / 'BSD'
    dst_dir = Path(project_root) / 'data' / 'bsd_denoise'
    
    noise_levels = [15, 25, 50]
    sample_dir = src_dir / 'BSD_noisy15' / 'input'
    
    if not sample_dir.exists():
        print(f"Directory not found: {sample_dir}. Skipping BSD.")
        return
        
    all_files = sorted(os.listdir(sample_dir))
    random.shuffle(all_files)
    
    num_val = int(len(all_files) * 0.20)
    val_files = all_files[:num_val]
    test_files = all_files[num_val:]
    
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    with open(dst_dir / 'val.txt', 'w') as f:
        for fname in val_files:
            f.write(fname + '\n')
            
    with open(dst_dir / 'test.txt', 'w') as f:
        for fname in test_files:
            f.write(fname + '\n')
            
    print(f"Generated splits: {len(val_files)} for val, {len(test_files)} for test.")
    
    for level in noise_levels:
        src_noise = src_dir / f'BSD_noisy{level}'
        dst_noise = dst_dir / f'noise{level}'
        
        dst_noisy_dir = dst_noise / 'noisy'
        dst_clean_dir = dst_noise / 'clean'
        
        dst_noisy_dir.mkdir(parents=True, exist_ok=True)
        dst_clean_dir.mkdir(parents=True, exist_ok=True)
        
        for fname in all_files:
            src_in = src_noise / 'input' / fname
            dst_in = dst_noisy_dir / fname
            if not dst_in.exists():
                shutil.copy2(src_in, dst_in)
                
            src_gt = src_noise / 'restored' / fname
            dst_gt = dst_clean_dir / fname
            if not dst_gt.exists():
                shutil.copy2(src_gt, dst_gt)
                
        print(f"Prepared noise{level} dataset.")

def prepare_reside(project_root):
    print("Preparing RESIDE-6K dataset...")
    random.seed(42)
    
    base_dir = Path(project_root) / "dataset" / "RESIDE-6K"
    test_dir = base_dir / "test"
    hazy_dir = test_dir / "hazy"
    gt_dir = test_dir / "GT"
    
    splits_dir = Path(project_root) / "data" / "reside6k" / "splits"
    
    if not hazy_dir.exists():
        print(f"Directory not found: {hazy_dir}. Skipping RESIDE.")
        return
        
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    hazy_images = sorted(list(hazy_dir.glob("*.jpg")) + list(hazy_dir.glob("*.png")))
    
    pairs = []
    for hazy_path in hazy_images:
        basename = hazy_path.name
        gt_path = gt_dir / basename
        
        if gt_path.exists():
            h_rel = hazy_path.relative_to(base_dir).as_posix()
            g_rel = gt_path.relative_to(base_dir).as_posix()
            pairs.append((h_rel, g_rel))
        else:
            print(f"Warning: Missing GT for {hazy_path}")
            
    print(f"Found {len(pairs)} pairs in RESIDE test folder.")
    
    random.shuffle(pairs)
    val_count = int(len(pairs) * 0.20)
    val_pairs = pairs[:val_count]
    test_pairs = pairs[val_count:]
    
    with open(splits_dir / "val.txt", "w") as f:
        for h, c in val_pairs:
            f.write(f"{h},{c}\n")
            
    with open(splits_dir / "test.txt", "w") as f:
        for h, c in test_pairs:
            f.write(f"{h},{c}\n")
            
    print(f"Saved {len(val_pairs)} to val.txt and {len(test_pairs)} to test.txt")

def prepare_dawn_fog(project_root):
    print("Preparing DAWN Fog dataset...")
    random.seed(42)
    
    base_dir = Path(project_root) / "dataset" / "DAWN" / "Fog"
    yolo_labels_dir = base_dir / "Fog_YOLO_darknet"
    
    splits_dir = Path(project_root) / "data" / "dawn" / "splits"
    std_images_dir = Path(project_root) / "data" / "dawn" / "images"
    std_labels_dir = Path(project_root) / "data" / "dawn" / "labels"
    
    if not base_dir.exists():
        print(f"Directory not found: {base_dir}. Skipping DAWN Fog.")
        return
        
    splits_dir.mkdir(parents=True, exist_ok=True)
    std_images_dir.mkdir(parents=True, exist_ok=True)
    std_labels_dir.mkdir(parents=True, exist_ok=True)
    
    images = sorted(list(base_dir.glob("*.jpg")))
    samples = []
    
    for img_path in images:
        basename = img_path.name
        label_name = img_path.stem + ".txt"
        label_path = yolo_labels_dir / label_name
        
        out_img_path = std_images_dir / basename
        out_lbl_path = std_labels_dir / label_name
        
        if not out_img_path.exists():
            shutil.copy2(img_path, out_img_path)
            
        if label_path.exists():
            with open(label_path, "r") as f_in, open(out_lbl_path, "w") as f_out:
                for line in f_in:
                    parts = line.strip().split()
                    if parts:
                        class_id = int(parts[0])
                        parts[0] = str(class_id - 1) # Convert 1-indexed to 0-indexed for YOLO
                        f_out.write(" ".join(parts) + "\n")
            samples.append((out_img_path.as_posix(), out_lbl_path.as_posix()))
        else:
            if not out_lbl_path.exists():
                open(out_lbl_path, 'w').close()
            samples.append((out_img_path.as_posix(), out_lbl_path.as_posix()))
            
    print(f"Found {len(samples)} fog images.")
    random.shuffle(samples)
    
    val_count = int(len(samples) * 0.20)
    val_samples = samples[:val_count]
    test_samples = samples[val_count:]
    
    with open(splits_dir / "fog_val.txt", "w") as f:
        for img, _ in val_samples:
            f.write(f"{img}\n")
            
    with open(splits_dir / "fog_test.txt", "w") as f:
        for img, _ in test_samples:
            f.write(f"{img}\n")
            
    with open(splits_dir / "fog_val_pairs.csv", "w") as f:
        for img, lbl in val_samples:
            f.write(f"{img},{lbl}\n")
            
    with open(splits_dir / "fog_test_pairs.csv", "w") as f:
        for img, lbl in test_samples:
            f.write(f"{img},{lbl}\n")
            
    print(f"Saved {len(val_samples)} to fog_val.txt and {len(test_samples)} to test.txt")

def prepare_lol(project_root):
    print("Preparing LOL dataset...")
    random.seed(42)
    
    source_dir = Path(project_root) / "dataset" / "LOL"
    dest_dir = Path(project_root) / "data" / "lol"
    
    our485_dir = source_dir / "our485"
    eval15_dir = source_dir / "eval15"
    
    if not our485_dir.exists() or not eval15_dir.exists():
        print(f"LOL dataset not found at {source_dir}. Skipping LOL.")
        return
        
    for split in ["train", "val", "test"]:
        for light in ["high", "low"]:
            (dest_dir / split / light).mkdir(parents=True, exist_ok=True)
            
    print("Preparing test split...")
    for light in ["high", "low"]:
        src_files = list((eval15_dir / light).glob("*.png"))
        for f in src_files:
            shutil.copy2(f, dest_dir / "test" / light / f.name)
            
    our485_files = sorted([f.name for f in (our485_dir / "high").glob("*.png")])
    random.shuffle(our485_files)
    train_files = our485_files[:400]
    val_files = our485_files[400:]
    
    print(f"Preparing train split ({len(train_files)} images)...")
    for f_name in train_files:
        shutil.copy2(our485_dir / "high" / f_name, dest_dir / "train" / "high" / f_name)
        shutil.copy2(our485_dir / "low" / f_name, dest_dir / "train" / "low" / f_name)
        
    print(f"Preparing val split ({len(val_files)} images)...")
    for f_name in val_files:
        shutil.copy2(our485_dir / "high" / f_name, dest_dir / "val" / "high" / f_name)
        shutil.copy2(our485_dir / "low" / f_name, dest_dir / "val" / "low" / f_name)
        
    print("LOL dataset preparation complete.")

def prepare_rain100h(project_root):
    print("Preparing rain100H dataset...")
    random.seed(42)
    
    base_dir = Path(project_root) / "dataset" / "rain100H"
    train_dir = base_dir / "train"
    rainy_dir = train_dir / "rain"
    clean_dir = train_dir / "norain"
    
    splits_dir = Path(project_root) / "data" / "rain100h" / "splits"
    
    if not rainy_dir.exists():
        print(f"Directory not found: {rainy_dir}. Skipping Rain100H.")
        return
        
    splits_dir.mkdir(parents=True, exist_ok=True)
    rainy_images = sorted(list(rainy_dir.glob("*.jpg")) + list(rainy_dir.glob("*.png")))
    
    pairs = []
    for rainy_path in rainy_images:
        basename = rainy_path.name
        clean_path = clean_dir / basename
        
        if not clean_path.exists():
            alt_basename = basename.replace("rain", "norain") if "norain" not in basename else basename.replace("norain", "rain")
            clean_path = clean_dir / alt_basename
            if not clean_path.exists():
                import re
                m = re.search(r'\d+', basename)
                if m:
                    idx = m.group(0)
                    for ext in [".png", ".jpg"]:
                        if (clean_dir / f"{idx}{ext}").exists():
                            clean_path = clean_dir / f"{idx}{ext}"
                            break
                        if (clean_dir / f"norain-{idx}{ext}").exists():
                            clean_path = clean_dir / f"norain-{idx}{ext}"
                            break
                            
        if clean_path.exists():
            r_rel = rainy_path.relative_to(base_dir).as_posix()
            c_rel = clean_path.relative_to(base_dir).as_posix()
            pairs.append((r_rel, c_rel))
        else:
            print(f"Warning: Missing clean image for {rainy_path}")
            
    print(f"Found {len(pairs)} pairs in rain100H test folder.")
    
    random.shuffle(pairs)
    val_count = int(len(pairs) * 0.20)
    val_pairs = pairs[:val_count]
    test_pairs = pairs[val_count:]
    
    with open(splits_dir / "val.txt", "w") as f:
        for r, c in val_pairs:
            f.write(f"{r},{c}\n")
            
    with open(splits_dir / "test.txt", "w") as f:
        for r, c in test_pairs:
            f.write(f"{r},{c}\n")
            
    print(f"Saved {len(val_pairs)} to val.txt and {len(test_pairs)} to test.txt")

def prepare_gopro(project_root, sample_size=1000):
    print("Preparing GoPro dataset...")
    random.seed(42)
    
    dataset_dir = Path(project_root) / 'dataset' / 'GoPro'
    output_csv = Path(project_root) / 'data' / 'gopro' / 'gopro_pairs.csv'
    
    input_dir = dataset_dir / 'input'
    sharp_dir = dataset_dir / 'restored'
    
    if dataset_dir.joinpath('target').exists():
        sharp_dir = dataset_dir / 'target'
    elif dataset_dir.joinpath('sharp').exists():
        sharp_dir = dataset_dir / 'sharp'
        
    if not input_dir.exists() or not sharp_dir.exists():
        print(f"Error: Missing input or sharp directory in {dataset_dir}. Skipping GoPro.")
        return
        
    input_files = set(f for f in os.listdir(input_dir) if f.endswith('.png'))
    sharp_files = set(f for f in os.listdir(sharp_dir) if f.endswith('.png'))
    
    common_files = list(input_files.intersection(sharp_files))
    print(f"Found {len(common_files)} matching pairs.")
    
    random.shuffle(common_files)
    if len(common_files) > sample_size:
        common_files = common_files[:sample_size]
        print(f"Subsampled to {sample_size} pairs.")
        
    num_val = int(len(common_files) * 0.2)
    val_files = common_files[:num_val]
    test_files = common_files[num_val:]
    
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['split', 'image_id', 'blur_path', 'sharp_path'])
        
        for img in val_files:
            writer.writerow(['val', img, (input_dir / img).as_posix(), (sharp_dir / img).as_posix()])
        for img in test_files:
            writer.writerow(['test', img, (input_dir / img).as_posix(), (sharp_dir / img).as_posix()])
            
    print(f"Saved pair index to {output_csv}")

def prepare_snow100k(project_root):
    print("Preparing Snow100K dataset...")
    random.seed(42)
    
    base_dir = Path(project_root) / "dataset" / "Snow100K"
    synthetic_dir = base_dir / "synthetic"
    gt_dir = base_dir / "gt"
    
    splits_dir = Path(project_root) / "data" / "snow100k" / "splits"
    
    if not synthetic_dir.exists():
        print(f"Directory not found: {synthetic_dir}. Skipping Snow100K.")
        return
        
    splits_dir.mkdir(parents=True, exist_ok=True)
    synthetic_images = sorted(list(synthetic_dir.glob("*.jpg")) + list(synthetic_dir.glob("*.png")))
    
    pairs = []
    for syn_path in synthetic_images:
        basename = syn_path.name
        gt_path = gt_dir / basename
        
        if gt_path.exists():
            syn_rel = syn_path.relative_to(base_dir).as_posix()
            gt_rel = gt_path.relative_to(base_dir).as_posix()
            pairs.append((syn_rel, gt_rel))
            
    print(f"Found {len(pairs)} pairs in Snow100K.")
    
    random.shuffle(pairs)
    # Subset to 1/10 to handle large dataset size
    sub_data = pairs[:len(pairs)//10]
    
    val_count = int(len(sub_data) * 0.20)
    val_pairs = sub_data[:val_count]
    test_pairs = sub_data[val_count:]
    
    with open(splits_dir / "val.txt", "w") as f:
        for s, c in val_pairs:
            f.write(f"{s},{c}\n")
            
    with open(splits_dir / "test.txt", "w") as f:
        for s, c in test_pairs:
            f.write(f"{s},{c}\n")
            
    print(f"Saved {len(val_pairs)} to val.txt and {len(test_pairs)} to test.txt")

def prepare_dawn_snow(project_root):
    print("Preparing DAWN Snow dataset...")
    random.seed(42)
    
    base_dataset_dir = Path(project_root) / "dataset" / "DAWN" / "Snow"
    xml_dir = base_dataset_dir / "Snow_PASCAL_VOC"
    
    out_base_dir = Path(project_root) / "data" / "dawn"
    out_images_dir = out_base_dir / "images" / "snow"
    out_labels_dir = out_base_dir / "labels_yolo" / "snow"
    splits_dir = out_base_dir / "splits"
    
    if not base_dataset_dir.exists():
        print(f"Directory not found: {base_dataset_dir}. Skipping DAWN Snow.")
        return
        
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    images = list(base_dataset_dir.glob("*.jpg"))
    samples = []
    
    for img_path in images:
        basename = img_path.name
        img_name = img_path.stem
        xml_path = xml_dir / (img_name + ".xml")
        
        out_img_path = out_images_dir / basename
        out_lbl_path = out_labels_dir / (img_name + ".txt")
        
        if not out_img_path.exists():
            shutil.copy2(img_path, out_img_path)
            
        if xml_path.exists():
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            size = root.find('size')
            w = int(size.find('width').text)
            h = int(size.find('height').text)
            
            with open(out_lbl_path, "w") as f_out:
                for obj in root.iter('object'):
                    cls = obj.find('name').text
                    if cls not in COCO_CLASS_TO_ID:
                        print(f"Warning: Unknown class '{cls}' in {xml_path}")
                        continue
                    cls_id = COCO_CLASS_TO_ID[cls]
                    xmlbox = obj.find('bndbox')
                    b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
                    bb = convert_to_yolo((w, h), b)
                    f_out.write(f"{cls_id} {' '.join(str(a) for a in bb)}\n")
                    
            samples.append((out_img_path.as_posix(), out_lbl_path.as_posix()))
        else:
            if not out_lbl_path.exists():
                open(out_lbl_path, 'w').close()
            samples.append((out_img_path.as_posix(), out_lbl_path.as_posix()))
            
    print(f"Prepared {len(samples)} DAWN Snow images.")
    random.shuffle(samples)
    
    val_count = int(len(samples) * 0.20)
    val_samples = samples[:val_count]
    test_samples = samples[val_count:]
    
    with open(splits_dir / "val_snow.txt", "w") as f:
        for img, _ in val_samples:
            f.write(f"{img}\n")
            
    with open(splits_dir / "test_snow.txt", "w") as f:
        for img, _ in test_samples:
            f.write(f"{img}\n")
            
    with open(splits_dir / "snow_val_pairs.csv", "w") as f:
        for img, lbl in val_samples:
            f.write(f"{img},{lbl}\n")
            
    with open(splits_dir / "snow_test_pairs.csv", "w") as f:
        for img, lbl in test_samples:
            f.write(f"{img},{lbl}\n")
            
    print(f"Saved {len(val_samples)} to val_snow.txt and {len(test_samples)} to test_snow.txt")
    
    yaml_path = out_base_dir / "dawn_snow.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {out_base_dir.resolve().as_posix()}\n")
        f.write("train: splits/test_snow.txt\n")
        f.write("val: splits/val_snow.txt\n")
        f.write("test: splits/test_snow.txt\n\n")
        f.write("names:\n")
        for k, v in COCO_CLASS_TO_ID.items():
            f.write(f"  {v}: {k}\n")
            
    print(f"Saved dataset yaml to {yaml_path}")

def prepare_dawn_sand(project_root):
    print("Preparing DAWN Sand dataset...")
    random.seed(42)
    
    base_dataset_dir = Path(project_root) / "dataset" / "DAWN" / "Sand"
    xml_dir = base_dataset_dir / "Sand_PASCAL_VOC"
    
    out_base_dir = Path(project_root) / "data" / "dawn"
    out_images_dir = out_base_dir / "images" / "sand"
    out_labels_dir = out_base_dir / "labels_yolo" / "sand"
    splits_dir = out_base_dir / "splits"
    
    if not base_dataset_dir.exists():
        print(f"Directory not found: {base_dataset_dir}. Skipping DAWN Sand.")
        return
        
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    images = list(base_dataset_dir.glob("*.jpg"))
    samples = []
    
    for img_path in images:
        basename = img_path.name
        img_name = img_path.stem
        xml_path = xml_dir / (img_name + ".xml")
        
        out_img_path = out_images_dir / basename
        out_lbl_path = out_labels_dir / (img_name + ".txt")
        
        if not out_img_path.exists():
            shutil.copy2(img_path, out_img_path)
            
        if xml_path.exists():
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            size = root.find('size')
            w = int(size.find('width').text)
            h = int(size.find('height').text)
            
            with open(out_lbl_path, "w") as f_out:
                for obj in root.iter('object'):
                    cls = obj.find('name').text
                    if cls not in COCO_CLASS_TO_ID:
                        print(f"Warning: Unknown class '{cls}' in {xml_path}")
                        continue
                    cls_id = COCO_CLASS_TO_ID[cls]
                    xmlbox = obj.find('bndbox')
                    b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
                    bb = convert_to_yolo((w, h), b)
                    f_out.write(f"{cls_id} {' '.join(str(a) for a in bb)}\n")
                    
            samples.append((out_img_path.as_posix(), out_lbl_path.as_posix()))
        else:
            if not out_lbl_path.exists():
                open(out_lbl_path, 'w').close()
            samples.append((out_img_path.as_posix(), out_lbl_path.as_posix()))
            
    print(f"Prepared {len(samples)} DAWN Sand images.")
    random.shuffle(samples)
    
    val_count = int(len(samples) * 0.20)
    val_samples = samples[:val_count]
    test_samples = samples[val_count:]
    
    with open(splits_dir / "val_sand.txt", "w") as f:
        for img, _ in val_samples:
            f.write(f"{img}\n")
            
    with open(splits_dir / "test_sand.txt", "w") as f:
        for img, _ in test_samples:
            f.write(f"{img}\n")
            
    with open(splits_dir / "sand_val_pairs.csv", "w") as f:
        for img, lbl in val_samples:
            f.write(f"{img},{lbl}\n")
            
    with open(splits_dir / "sand_test_pairs.csv", "w") as f:
        for img, lbl in test_samples:
            f.write(f"{img},{lbl}\n")
            
    print(f"Saved {len(val_samples)} to val_sand.txt and {len(test_samples)} to test_sand.txt")
    
    yaml_path = out_base_dir / "dawn_sand.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {out_base_dir.resolve().as_posix()}\n")
        f.write("train: splits/val_sand.txt\n")
        f.write("val: splits/val_sand.txt\n")
        f.write("test: splits/test_sand.txt\n\n")
        f.write("names:\n")
        for k, v in sorted(COCO_CLASS_TO_ID.items(), key=lambda item: item[1]):
            f.write(f"  {v}: {k}\n")
            
    print(f"Saved dataset yaml to {yaml_path}")

def prepare_dawn_rain(project_root):
    print("Preparing DAWN Rain dataset...")
    random.seed(42)
    
    base_dir = Path(project_root) / "dataset" / "DAWN" / "Rain"
    yolo_labels_dir = base_dir / "Rain_YOLO_darknet"
    
    splits_dir = Path(project_root) / "data" / "dawn_rain" / "splits"
    std_images_dir = Path(project_root) / "data" / "dawn_rain" / "images"
    std_labels_dir = Path(project_root) / "data" / "dawn_rain" / "labels"
    
    if not base_dir.exists():
        print(f"Directory not found: {base_dir}. Skipping DAWN Rain.")
        return
        
    splits_dir.mkdir(parents=True, exist_ok=True)
    std_images_dir.mkdir(parents=True, exist_ok=True)
    std_labels_dir.mkdir(parents=True, exist_ok=True)
    
    images = sorted(list(base_dir.glob("*.jpg")))
    samples = []
    
    for img_path in images:
        basename = img_path.name
        label_name = img_path.stem + ".txt"
        label_path = yolo_labels_dir / label_name
        
        out_img_path = std_images_dir / basename
        out_lbl_path = std_labels_dir / label_name
        
        if not out_img_path.exists():
            shutil.copy2(img_path, out_img_path)
            
        if label_path.exists():
            with open(label_path, "r") as f_in, open(out_lbl_path, "w") as f_out:
                for line in f_in:
                    parts = line.strip().split()
                    if parts:
                        class_id = int(parts[0])
                        parts[0] = str(class_id - 1)
                        f_out.write(" ".join(parts) + "\n")
            samples.append((out_img_path.as_posix(), out_lbl_path.as_posix()))
        else:
            if not out_lbl_path.exists():
                open(out_lbl_path, 'w').close()
            samples.append((out_img_path.as_posix(), out_lbl_path.as_posix()))
            
    print(f"Found {len(samples)} rain images.")
    random.shuffle(samples)
    
    val_count = int(len(samples) * 0.20)
    val_samples = samples[:val_count]
    test_samples = samples[val_count:]
    
    with open(splits_dir / "rain_val.txt", "w") as f:
        for img, _ in val_samples:
            f.write(f"{img}\n")
            
    with open(splits_dir / "rain_test.txt", "w") as f:
        for img, _ in test_samples:
            f.write(f"{img}\n")
            
    with open(splits_dir / "rain_val_pairs.csv", "w") as f:
        for img, lbl in val_samples:
            f.write(f"{img},{lbl}\n")
            
    with open(splits_dir / "rain_test_pairs.csv", "w") as f:
        for img, lbl in test_samples:
            f.write(f"{img},{lbl}\n")
            
    print(f"Saved {len(val_samples)} to rain_val.txt and {len(test_samples)} to test.txt")

def prepare_foggycityscapes(project_root):
    print("Preparing Foggy Cityscapes dataset...")
    base_dir = Path(project_root) / "dataset" / "Foggy_Cityscapes"
    
    if not base_dir.exists():
        print(f"Directory not found: {base_dir}. Skipping Foggy Cityscapes.")
        return
        
    images = list(base_dir.glob("**/images/*.jpg")) + list(base_dir.glob("**/images/*.png"))
    pairs = []
    
    for img_path in images:
        lbl_path = Path(str(img_path).replace("images", "labels").replace(".jpg", ".txt").replace(".png", ".txt"))
        if lbl_path.exists():
            pairs.append((img_path.as_posix(), lbl_path.as_posix()))
            
    pairs = get_subset(pairs)
    split_dir = Path(project_root) / "data" / "foggycityscapes" / "splits"
    split_and_save_pairs(pairs, split_dir)

def prepare_rtts(project_root):
    print("Preparing RTTS dataset...")
    base_dir = Path(project_root) / "dataset" / "RTTS" / "RTTS"
    img_dir = base_dir / "JPEGImages"
    xml_dir = base_dir / "Annotations"
    lbl_dir = base_dir / "labels"
    
    if not base_dir.exists():
        print(f"Directory not found: {base_dir}. Skipping RTTS.")
        return
        
    lbl_dir.mkdir(parents=True, exist_ok=True)
    xml_files = list(xml_dir.glob("*.xml"))
    pairs = []
    
    for xml_file in xml_files:
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            size = root.find("size")
            width = float(size.find("width").text)
            height = float(size.find("height").text)
            
            if width == 0 or height == 0: continue
            
            yolo_lines = []
            for obj in root.findall("object"):
                cls_name = obj.find("name").text.lower()
                cls_mapped = "motorcycle" if cls_name == "motorbike" else cls_name
                if cls_mapped not in COCO_CLASS_TO_ID:
                    continue
                cls_id = COCO_CLASS_TO_ID[cls_mapped]
                
                bndbox = obj.find("bndbox")
                xmin = float(bndbox.find("xmin").text)
                ymin = float(bndbox.find("ymin").text)
                xmax = float(bndbox.find("xmax").text)
                ymax = float(bndbox.find("ymax").text)
                
                cx = (xmin + xmax) / 2.0 / width
                cy = (ymin + ymax) / 2.0 / height
                bw = (xmax - xmin) / width
                bh = (ymax - ymin) / height
                
                yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                
            img_name = xml_file.stem + ".png"
            img_path = img_dir / img_name
            
            if not img_path.exists():
                img_name = xml_file.stem + ".jpg"
                img_path = img_dir / img_name
                
            if img_path.exists():
                lbl_path = lbl_dir / (xml_file.stem + ".txt")
                with open(lbl_path, "w") as f:
                    f.write("\n".join(yolo_lines) + "\n")
                pairs.append((img_path.as_posix(), lbl_path.as_posix()))
        except Exception:
            pass
            
    pairs = get_subset(pairs)
    split_dir = Path(project_root) / "data" / "rtts" / "splits"
    split_and_save_pairs(pairs, split_dir)

# -------------------------------------------------------------
# Main Execution Entrypoint
# -------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Consolidated dataset preparation helper for ProjectCV.")
    parser.add_argument(
        "--project-root",
        type=str,
        default=str(get_project_root()),
        help="Path to the root directory of the project."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=[
            "all", "bsd", "reside", "dawn_fog", "lol", "rain100h",
            "gopro", "snow100k", "dawn_snow", "dawn_sand", "dawn_rain",
            "foggycityscapes", "rtts"
        ],
        help="Specific dataset to prepare, or 'all' to prepare everything."
    )
    parser.add_argument(
        "--gopro-sample-size",
        type=int,
        default=1000,
        help="Sample size limit for GoPro dataset processing."
    )
    args = parser.parse_args()
    
    project_root = args.project_root
    dataset = args.dataset
    
    print(f"Using Project Root: {project_root}")
    
    if dataset in ["all", "bsd"]:
        prepare_bsd(project_root)
    if dataset in ["all", "reside"]:
        prepare_reside(project_root)
    if dataset in ["all", "dawn_fog"]:
        prepare_dawn_fog(project_root)
    if dataset in ["all", "lol"]:
        prepare_lol(project_root)
    if dataset in ["all", "rain100h"]:
        prepare_rain100h(project_root)
    if dataset in ["all", "gopro"]:
        prepare_gopro(project_root, args.gopro_sample_size)
    if dataset in ["all", "snow100k"]:
        prepare_snow100k(project_root)
    if dataset in ["all", "dawn_snow"]:
        prepare_dawn_snow(project_root)
    if dataset in ["all", "dawn_sand"]:
        prepare_dawn_sand(project_root)
    if dataset in ["all", "dawn_rain"]:
        prepare_dawn_rain(project_root)
    if dataset in ["all", "foggycityscapes"]:
        prepare_foggycityscapes(project_root)
    if dataset in ["all", "rtts"]:
        prepare_rtts(project_root)

if __name__ == "__main__":
    main()
