import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
DL_NETS_DIR = ROOT / "dl_nets"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(DL_NETS_DIR))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, to_pil_image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def import_adair(adair_repo: Path):
    sys.path.insert(0, str(adair_repo))
    from net.model import AdaIR
    return AdaIR


def load_adair_checkpoint(model: torch.nn.Module, ckpt_path: Path) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    cleaned = {}
    for key, value in state.items():
        if key.startswith("net."):
            key = key[4:]
        cleaned[key] = value
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing or unexpected:
        print(f"AdaIR load_state_dict missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print("missing sample:", missing[:10])
        if unexpected:
            print("unexpected sample:", unexpected[:10])


def pad_to_multiple(x: torch.Tensor, multiple: int = 32) -> tuple[torch.Tensor, int, int]:
    h, w = x.shape[-2:]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return x, h, w


def restore_tiled(model: torch.nn.Module, x: torch.Tensor, tile: int, overlap: int) -> torch.Tensor:
    b, _, h, w = x.shape
    if b != 1:
        raise ValueError("Tiled inference supports batch size 1 only.")
    stride = max(tile - overlap, 1)
    out = torch.zeros_like(x)
    weight = torch.zeros_like(x)
    for top in range(0, h, stride):
        for left in range(0, w, stride):
            bottom = min(top + tile, h)
            right = min(left + tile, w)
            top0 = max(bottom - tile, 0)
            left0 = max(right - tile, 0)
            patch = x[:, :, top0:bottom, left0:right]
            patch, patch_h, patch_w = pad_to_multiple(patch, 32)
            restored = model(patch).clamp(0, 1)[:, :, :patch_h, :patch_w]
            out[:, :, top0:bottom, left0:right] += restored
            weight[:, :, top0:bottom, left0:right] += 1.0
    return out / weight.clamp_min(1.0)


def restore_image(model: torch.nn.Module, image_path: Path, out_path: Path, device: torch.device, tile: int, overlap: int) -> None:
    img = Image.open(image_path).convert("RGB")
    x = pil_to_tensor(img).float().unsqueeze(0) / 255.0
    x, h, w = pad_to_multiple(x, 32)
    x = x.to(device)
    with torch.no_grad():
        y = restore_tiled(model, x, tile, overlap) if tile > 0 else model(x).clamp(0, 1)
    y = y[:, :, :h, :w].cpu().squeeze(0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    to_pil_image(y).save(out_path)


def discover_images(paths: list[str], max_images: int) -> list[Path]:
    images: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            print(f"skip missing input path: {path}")
            continue
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            images.append(path)
        elif path.is_dir():
            images.extend(sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS))
    # Avoid YOLO label/annotation folders accidentally containing preview images.
    images = [p for p in images if not any(part.endswith("_YOLO_darknet") or part.endswith("_PASCAL_VOC") for part in p.parts)]
    if max_images > 0:
        images = images[:max_images]
    return images


def result_summary(result, class_names: dict[int, str]) -> dict[str, object]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return {
            "det_count": 0,
            "mean_conf": 0.0,
            "max_conf": 0.0,
            "class_counts": "{}",
        }
    conf = boxes.conf.detach().cpu().numpy()
    cls = boxes.cls.detach().cpu().numpy().astype(int)
    counts: dict[str, int] = {}
    for c in cls:
        counts[class_names.get(int(c), str(int(c)))] = counts.get(class_names.get(int(c), str(int(c))), 0) + 1
    return {
        "det_count": int(len(boxes)),
        "mean_conf": float(conf.mean()) if len(conf) else 0.0,
        "max_conf": float(conf.max()) if len(conf) else 0.0,
        "class_counts": str(counts),
    }


def save_side_by_side(original_annotated: np.ndarray, restored_annotated: np.ndarray, out_path: Path) -> None:
    h = max(original_annotated.shape[0], restored_annotated.shape[0])
    def pad(img):
        if img.shape[0] == h:
            return img
        pad_h = h - img.shape[0]
        return cv2.copyMakeBorder(img, 0, pad_h, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    side = np.concatenate([pad(original_annotated), pad(restored_annotated)], axis=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), side)


def safe_rel(path: Path, roots: list[Path]) -> Path:
    for root in roots:
        try:
            return path.relative_to(root)
        except ValueError:
            continue
    return Path(path.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Restore RTTS/DAWN images with AdaIR, then compare YOLO detections before/after restoration.")
    parser.add_argument("--adair-root", "--adair-repo", dest="adair_repo", default=str(ROOT / "dl_nets" / "AdaIR"))
    parser.add_argument("--ckpt", "--adair-ckpt", dest="adair_ckpt", default=str(ROOT / "checkpoints" / "adair" / "adair5d.ckpt"))
    parser.add_argument("--input-dirs", nargs="+", default=["data/DAWN/Fog", "data/DAWN/Rain", "data/DAWN/Snow", "data/DAWN/Sand", "data/RTTS"])
    parser.add_argument("--output-dir", default=str(ROOT / "results" / "adair" / "restoration_yolo" / "yolo_compare"))
    parser.add_argument("--yolo-weights", default="yolov8n.pt", help="YOLO weights/model name. Use your local .pt path if available.")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--tile", type=int, default=256)
    parser.add_argument("--tile-overlap", type=int, default=32)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--restore-only", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    restored_dir = out_dir / "restored"
    yolo_orig_dir = out_dir / "yolo_original"
    yolo_restored_dir = out_dir / "yolo_restored"
    compare_dir = out_dir / "compare"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    AdaIR = import_adair(Path(args.adair_repo).resolve())
    adair = AdaIR(decoder=True)
    load_adair_checkpoint(adair, Path(args.adair_ckpt))
    adair.to(device).eval()

    input_roots = [Path(p).resolve() for p in args.input_dirs if Path(p).exists()]
    images = discover_images(args.input_dirs, args.max_images)
    if not images:
        raise RuntimeError(f"No images found in input dirs: {args.input_dirs}")
    print(f"device={device}")
    print(f"images={len(images)}")
    print(f"adair_ckpt={args.adair_ckpt}")
    print(f"output_dir={out_dir}")

    restored_paths: list[tuple[Path, Path]] = []
    for idx, image_path in enumerate(images, 1):
        rel = safe_rel(image_path.resolve(), input_roots)
        restored_path = restored_dir / rel.with_suffix(".png")
        print(f"[restore {idx}/{len(images)}] {image_path}")
        restore_image(adair, image_path, restored_path, device, args.tile, args.tile_overlap)
        restored_paths.append((image_path, restored_path))

    if args.restore_only:
        return

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "ultralytics is not installed. Install it first with: pip install ultralytics"
        ) from exc

    yolo = YOLO(args.yolo_weights)
    class_names = yolo.names if isinstance(yolo.names, dict) else {i: name for i, name in enumerate(yolo.names)}
    rows: list[dict[str, object]] = []
    for idx, (original_path, restored_path) in enumerate(restored_paths, 1):
        rel = safe_rel(original_path.resolve(), input_roots)
        print(f"[yolo {idx}/{len(restored_paths)}] {original_path.name}")
        orig_result = yolo.predict(str(original_path), imgsz=args.imgsz, conf=args.conf, iou=args.iou, verbose=False)[0]
        rest_result = yolo.predict(str(restored_path), imgsz=args.imgsz, conf=args.conf, iou=args.iou, verbose=False)[0]

        orig_ann = orig_result.plot()
        rest_ann = rest_result.plot()
        orig_out = yolo_orig_dir / rel.with_suffix(".jpg")
        rest_out = yolo_restored_dir / rel.with_suffix(".jpg")
        comp_out = compare_dir / rel.with_suffix(".jpg")
        orig_out.parent.mkdir(parents=True, exist_ok=True)
        rest_out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(orig_out), orig_ann)
        cv2.imwrite(str(rest_out), rest_ann)
        save_side_by_side(orig_ann, rest_ann, comp_out)

        orig_sum = result_summary(orig_result, class_names)
        rest_sum = result_summary(rest_result, class_names)
        rows.append({
            "image_path": str(original_path),
            "restored_path": str(restored_path),
            "original_det_count": orig_sum["det_count"],
            "restored_det_count": rest_sum["det_count"],
            "delta_det_count": int(rest_sum["det_count"]) - int(orig_sum["det_count"]),
            "original_mean_conf": orig_sum["mean_conf"],
            "restored_mean_conf": rest_sum["mean_conf"],
            "delta_mean_conf": float(rest_sum["mean_conf"]) - float(orig_sum["mean_conf"]),
            "original_max_conf": orig_sum["max_conf"],
            "restored_max_conf": rest_sum["max_conf"],
            "original_class_counts": orig_sum["class_counts"],
            "restored_class_counts": rest_sum["class_counts"],
            "original_annotated": str(orig_out),
            "restored_annotated": str(rest_out),
            "comparison": str(comp_out),
        })

    metrics_path = out_dir / "yolo_before_after_summary.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"metrics={metrics_path}")
    print(f"comparisons={compare_dir}")


if __name__ == "__main__":
    main()
