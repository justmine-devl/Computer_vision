import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
DL_NETS_DIR = ROOT / "dl_nets"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(DL_NETS_DIR))

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from torchvision.utils import make_grid


def import_adair(adair_repo: Path):
    sys.path.insert(0, str(adair_repo))
    from net.model import AdaIR
    return AdaIR


def load_pretrained(model: torch.nn.Module, ckpt_path: Path) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    cleaned = {}
    for key, value in state.items():
        if key.startswith("net."):
            key = key[4:]
        cleaned[key] = value
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing or unexpected:
        print(f"load_state_dict missing={len(missing)} unexpected={len(unexpected)}")
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


def infer_tiled(model: torch.nn.Module, x: torch.Tensor, tile: int, overlap: int) -> torch.Tensor:
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
            patch, patch_h, patch_w = pad_to_multiple(patch, multiple=32)
            restored = model(patch).clamp(0, 1)[:, :, :patch_h, :patch_w]
            out[:, :, top0:bottom, left0:right] += restored
            weight[:, :, top0:bottom, left0:right] += 1.0
    return out / weight.clamp_min(1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adair-root", "--adair-repo", dest="adair_repo", default=str(ROOT / "dl_nets" / "AdaIR"))
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--input-image", "--input", dest="input", required=True)
    parser.add_argument("--output", default=str(ROOT / "results" / "adair" / "restoration_yolo" / "single_restored.png"))
    parser.add_argument("--comparison", default="")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--tile", type=int, default=0, help="Use tiled inference with this tile size, e.g. 256.")
    parser.add_argument("--tile-overlap", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    AdaIR = import_adair(Path(args.adair_repo).resolve())
    model = AdaIR(decoder=True)
    load_pretrained(model, Path(args.ckpt))
    model.to(device).eval()

    img = Image.open(args.input).convert("RGB")
    x = pil_to_tensor(img).float().unsqueeze(0) / 255.0
    x, h, w = pad_to_multiple(x, multiple=32)
    x = x.to(device)

    with torch.no_grad():
        if args.tile > 0:
            y = infer_tiled(model, x, tile=args.tile, overlap=args.tile_overlap)
        else:
            y = model(x).clamp(0, 1)
    y = y[:, :, :h, :w].cpu()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    to_pil_image(y.squeeze(0)).save(output_path)

    if args.comparison:
        comparison_path = Path(args.comparison)
        comparison_path.parent.mkdir(parents=True, exist_ok=True)
        original = pil_to_tensor(img).float() / 255.0
        grid = make_grid(torch.stack([original, y.squeeze(0)], dim=0), nrow=2)
        to_pil_image(grid).save(comparison_path)

    print(f"device={device}")
    print(f"input={args.input}")
    print(f"output={output_path}")
    if args.comparison:
        print(f"comparison={args.comparison}")


if __name__ == "__main__":
    main()
