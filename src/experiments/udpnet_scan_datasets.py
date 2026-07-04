#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
DL_NETS_DIR = ROOT / "dl_nets"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(DL_NETS_DIR))

from pipelines.udpnet_pipeline.common.cli import add_common_path_args, apply_common_path_overrides
from pipelines.udpnet_pipeline.common.config import load_yaml_config
from pipelines.udpnet_pipeline.data.scan_and_pair import scan_and_pair_datasets

def parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(
                description=(
                        "Scan configured datasets (DAWN/RTTS/...) and organize image-label pairs "
                        "into a unified structure for detection training/evaluation."
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
                help="Optional subset of datasets to scan. Example: --datasets DAWN RTTS",
        )
        parser.add_argument(
                "--link-mode",
                choices=["symlink", "hardlink", "copy"],
                default=None,
                help="Override link mode from config.",
        )
        parser.add_argument(
                "--overwrite",
                action="store_true",
                help="Overwrite existing organized files.",
        )
        parser.add_argument(
                "--dry-run",
                action="store_true",
                help="Scan only. No files/manifests written.",
        )
        add_common_path_args(parser)
        return parser.parse_args()


def main() -> int:
        args = parse_args()

        config_path = Path(args.config).expanduser().resolve()
        config = load_yaml_config(config_path)
        apply_common_path_overrides(config, args)

        overwrite_value: Optional[bool] = True if args.overwrite else None
        selected: Optional[Sequence[str]] = args.datasets if args.datasets else None

        summary = scan_and_pair_datasets(
                config=config,
                selected_datasets=selected,
                link_mode=args.link_mode,
                overwrite=overwrite_value,
                dry_run=args.dry_run,
        )

        print(json.dumps(summary, indent=2))
        return 0


if __name__ == "__main__":
        raise SystemExit(main())

