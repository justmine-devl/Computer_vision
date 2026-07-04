from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rename Rain100L target files from norain* to rain*.")
    parser.add_argument("--target-dir", required=True, help="Directory containing files to rename.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned changes without renaming files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    target_dir = Path(args.target_dir)
    if not target_dir.is_dir():
        raise FileNotFoundError(f"Target directory does not exist: {target_dir}")

    renamed = 0
    skipped = 0
    for path in sorted(target_dir.iterdir()):
        if not path.is_file() or "norain" not in path.name:
            continue

        new_path = path.with_name(path.name.replace("norain", "rain"))
        if new_path.exists():
            skipped += 1
            print(f"skip exists: {new_path.name}")
            continue

        print(f"{path.name} -> {new_path.name}")
        if not args.dry_run:
            path.rename(new_path)
        renamed += 1

    print(f"renamed={renamed}, skipped={skipped}, dry_run={args.dry_run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
