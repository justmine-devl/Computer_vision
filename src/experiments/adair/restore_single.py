import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from torchvision.utils import make_grid

from utils.adair import import_adair, load_adair_checkpoint, pad_to_multiple, restore_image, restore_tensor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adair-root", "--adair-repo", dest="adair_repo", default=str(ROOT / "dl_nets" / "AdaIR"))
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--input-image", "--input", dest="input", required=True)
    parser.add_argument("--output", default=str(ROOT / "outputs" / "adair" / "restore_single" / "single_restored.png"))
    parser.add_argument("--comparison", default="")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--tile", type=int, default=0, help="Use tiled inference with this tile size, e.g. 256.")
    parser.add_argument("--tile-overlap", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    AdaIR = import_adair(Path(args.adair_repo).resolve())
    model = AdaIR(decoder=True)
    load_adair_checkpoint(model, Path(args.ckpt))
    model.to(device).eval()

    output_path = Path(args.output)
    y = restore_image(model, Path(args.input), output_path, device, tile=args.tile, overlap=args.tile_overlap)

    if args.comparison:
        comparison_path = Path(args.comparison)
        comparison_path.parent.mkdir(parents=True, exist_ok=True)
        img = Image.open(args.input).convert("RGB")
        original = pil_to_tensor(img).float() / 255.0
        restored = y.squeeze(0)
        grid = make_grid(torch.stack([original, restored], dim=0), nrow=2)
        to_pil_image(grid).save(comparison_path)

    print(f"device={device}")
    print(f"input={args.input}")
    print(f"output={output_path}")
    if args.comparison:
        print(f"comparison={args.comparison}")


if __name__ == "__main__":
    main()
