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
from pipelines.udpnet.depth.generate_depthmaps import generate_depthmaps_for_datasets

def parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(
                description=(
                        "Generate depth maps for organized datasets by calling the bundled UDPNet depth helper."
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
                help="Optional subset. Example: --datasets DAWN RTTS",
        )
        parser.add_argument(
                "--overwrite",
                action="store_true",
                help="Regenerate even if output directory exists.",
        )
        parser.add_argument(
                "--dry-run",
                action="store_true",
                help="Plan only; do not run depth generation.",
        )
        parser.add_argument("--depth-weights", type=str, default=None, help="DepthAnything weights override.")
        add_common_path_args(parser)
        return parser.parse_args()


def main() -> int:
        args = parse_args()
        config = load_yaml_config(Path(args.config).expanduser().resolve())
        apply_common_path_overrides(config, args)

        selected: Optional[Sequence[str]] = args.datasets if args.datasets else None

        summary = generate_depthmaps_for_datasets(
                config=config,
                selected_datasets=selected,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
        )

        print(json.dumps(summary, indent=2))
        return 0


if __name__ == "__main__":
        raise SystemExit(main())

