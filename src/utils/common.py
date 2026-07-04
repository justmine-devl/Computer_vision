from __future__ import annotations

import sys
from pathlib import Path


def setup_project_paths(file_path: str | Path) -> Path:
    """Add repo src/ and dl_nets/ to sys.path for scripts run from the repo root."""
    root = Path(file_path).resolve().parents[2]
    src_dir = root / "src"
    dl_nets_dir = root / "dl_nets"
    for path in (src_dir, dl_nets_dir):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
    return root
