from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

ConfigDict = Dict[str, Any]


def add_common_path_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--data-root", type=str, default=None, help="Dataset root override.")
        parser.add_argument("--ckpt", type=str, default=None, help="UDPNet checkpoint override.")
        parser.add_argument("--output-dir", type=str, default=None, help="Output directory override.")
        parser.add_argument("--method-root", type=str, default=None, help="UDPNet method root override.")
        parser.add_argument("--yolo-weights", type=str, default=None, help="YOLO weights override.")
        parser.add_argument("--device", type=str, default=None, help="Runtime device override.")


def apply_common_path_overrides(config: ConfigDict, args: argparse.Namespace) -> ConfigDict:
        paths_cfg = config.setdefault("paths", {})
        runtime_cfg = config.setdefault("runtime", {})
        restoration_cfg = config.setdefault("restoration", {})
        detection_cfg = config.setdefault("detection", {})

        data_root = getattr(args, "data_root", None)
        if data_root:
                paths_cfg["datasets_root"] = data_root
                paths_cfg.setdefault("organized_root", str(Path(data_root) / "organized" / "UDPNet"))
                for name, entry in config.get("datasets", {}).get("entries", {}).items():
                        if isinstance(entry, dict):
                                entry["root"] = str(Path(data_root) / str(name))

        output_dir = getattr(args, "output_dir", None)
        if output_dir:
                paths_cfg["outputs_root"] = output_dir

        ckpt = getattr(args, "ckpt", None)
        if ckpt:
                restoration_cfg["checkpoint"] = ckpt

        method_root = getattr(args, "method_root", None)
        if method_root:
                restoration_cfg["method_root"] = method_root

        yolo_weights = getattr(args, "yolo_weights", None)
        if yolo_weights:
                detection_cfg["weights"] = yolo_weights

        device = getattr(args, "device", None)
        if device:
                runtime_cfg["device"] = device

        depth_weights = getattr(args, "depth_weights", None)
        if depth_weights:
                config.setdefault("depth", {})["weights"] = depth_weights

        return config
