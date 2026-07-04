from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
DL_NETS_DIR = ROOT / "dl_nets"


def setup_project_path(include_dl_nets: bool = True) -> Path:
    paths = [ROOT, SRC_DIR]
    if include_dl_nets:
        paths.append(DL_NETS_DIR)
    for path in paths:
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
    return ROOT
