from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
DL_NETS_DIR = ROOT / "dl_nets"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(DL_NETS_DIR))

from hưng.src.utils.udpnet_pipeline.common.cli import apply_common_path_overrides
from hưng.src.utils.udpnet_pipeline.common.config import load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal UDPNet training entrypoint.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "src" / "training" / "udpnet_pipeline.yaml",
        help="UDPNet pipeline config.",
    )
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "UDPNet" / "train")
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--method-root", type=Path, default=DL_NETS_DIR / "UDPNet")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model", choices=["FSNet_UDPNet", "ConvIR_UDPNet"], default="FSNet_UDPNet")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=7)
    return parser.parse_args()


def build_model(model_name: str):
    if model_name == "FSNet_UDPNet":
        from UDPNet.models.FSNet_UDPNet import build_net
    else:
        from UDPNet.models.ConvIR_UDPNet import build_net

    return build_net()


def main() -> int:
    args = parse_args()
    config = load_yaml_config(args.config)
    apply_common_path_overrides(config, args)
    model = build_model(args.model)

    print(f"model={type(model).__name__}")
    print(f"config={args.config}")
    print(f"config_project={config.get('project', {}).get('name')}")
    print(f"data_root={args.data_root}")
    print(f"output_dir={args.output_dir}")
    print("Training loop not migrated: provide real data/checkpoint first, then port OTS trainlightning minimally.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
