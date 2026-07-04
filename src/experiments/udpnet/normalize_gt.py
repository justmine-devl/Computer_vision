#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = ROOT / "src"
DL_NETS_DIR = ROOT / "dl_nets"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(DL_NETS_DIR))

from pipelines.udpnet.common.cli import add_common_path_args, apply_common_path_overrides
from pipelines.udpnet.common.config import load_yaml_config
from pipelines.udpnet.data.gt_normalize import normalize_ground_truth

def parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(
                description=(
                        "Normalize DAWN/RTTS (and future datasets) annotations into YOLO txt format "
                        "using config-driven converters and class maps."
                )
        )
        parser.add_argument(
                "--config",
                type=str,
                default=str(ROOT / "src" / "training" / "udpnet_pipeline.yaml"),
                help="Path to YAML config.",
        )
        parser.add_argument(
                "--datasets",
                nargs="*",
                default=None,
                help="Optional subset of datasets. Example: --datasets DAWN RTTS",
        )
        parser.add_argument(
                "--overwrite",
                action="store_true",
                help="Overwrite existing normalized labels.",
        )
        add_common_path_args(parser)
        return parser.parse_args()


def main() -> int:
        args = parse_args()
        config_path = Path(args.config).expanduser().resolve()
        config = load_yaml_config(config_path)
        apply_common_path_overrides(config, args)

        selected: Optional[Sequence[str]] = args.datasets if args.datasets else None

        summary = normalize_ground_truth(
                config=config,
                selected_datasets=selected,
                overwrite=args.overwrite,
        )

        print(json.dumps(summary, indent=2))
        return 0


if __name__ == "__main__":
        raise SystemExit(main())

