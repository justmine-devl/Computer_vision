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
from pipelines.udpnet_pipeline.run.evaluate_pipeline import evaluate_pipeline

def parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(
                description="Run end-to-end restoration -> detection -> IoU/mAP evaluation pipeline."
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
                "--max-images",
                type=int,
                default=None,
                help="Optional hard cap on number of images to process.",
        )
        parser.add_argument(
                "--run-name",
                type=str,
                default=None,
                help="Optional label recorded in metrics.json; output dir is auto-derived.",
        )
        parser.add_argument(
                "--dry-run",
                action="store_true",
                help="Initialize components and data only; skip model inference.",
        )
        parser.add_argument(
                "--detector-input-mode",
                choices=["file_path", "tensor"],
                default=None,
                help="Override evaluation.detector_input_mode.",
        )
        parser.add_argument(
                "--detection-imgsz",
                type=int,
                default=None,
                help="Override detection.imgsz for YOLO inference.",
        )
        parser.add_argument(
                "--save-restored-full-resolution",
                action="store_true",
                help="Save restored images resized back to original dimensions.",
        )
        parser.add_argument(
                "--metrics-parity",
                choices=["rtts_script"],
                default=None,
                help="Apply external RTTS parity metric settings.",
        )
        add_common_path_args(parser)
        return parser.parse_args()


def main() -> int:
        args = parse_args()
        config = load_yaml_config(Path(args.config).expanduser().resolve())
        apply_common_path_overrides(config, args)

        if args.detector_input_mode is not None:
                config.setdefault("evaluation", {})[
                        "detector_input_mode"
                ] = args.detector_input_mode
        if args.detection_imgsz is not None:
                config.setdefault("detection", {})["imgsz"] = int(
                        args.detection_imgsz
                )
        if args.save_restored_full_resolution:
                config.setdefault("evaluation", {})[
                        "save_restored_full_resolution"
                ] = True
        if args.metrics_parity == "rtts_script":
                evaluation_cfg = config.setdefault("evaluation", {})
                evaluation_cfg["detector_input_mode"] = "file_path"
                evaluation_cfg["map_class_set"] = "gt_union_pred"
                evaluation_cfg["iou_primary"] = "all_gt"
                evaluation_cfg["save_restored_full_resolution"] = True

        selected: Optional[Sequence[str]] = args.datasets if args.datasets else None

        summary = evaluate_pipeline(
                config=config,
                selected_datasets=selected,
                max_images=args.max_images,
                run_name=args.run_name,
                dry_run=args.dry_run,
        )

        print(json.dumps(summary, indent=2))
        return 0


if __name__ == "__main__":
        raise SystemExit(main())

