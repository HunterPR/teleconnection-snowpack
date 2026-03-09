"""
Storage cleanup helper. Run from repo root.

Usage:
  python scripts/cleanup_storage.py              # remove __pycache__ only
  python scripts/cleanup_storage.py --pipeline  # also remove data/pipeline/*
  python scripts/cleanup_storage.py --dry-run   # print what would be removed
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(description="Clean caches and optional generated data.")
    ap.add_argument("--dry-run", action="store_true", help="Only print actions")
    ap.add_argument("--pipeline", action="store_true", help="Also remove data/pipeline/*")
    ap.add_argument("--processed", action="store_true", help="Also remove data/processed/* (regenerate with organize_data.py)")
    args = ap.parse_args()

    removed: list[str] = []

    for dirpath, dirnames, _ in os.walk(ROOT, topdown=True):
        if "__pycache__" in dirnames:
            p = Path(dirpath) / "__pycache__"
            rel = p.relative_to(ROOT)
            if args.dry_run:
                print(f"[dry-run] would remove: {rel}")
            else:
                try:
                    shutil.rmtree(p)
                    removed.append(str(rel))
                except Exception as e:
                    print(f"Skip {rel}: {e}")
            dirnames[:] = [d for d in dirnames if d != "__pycache__"]

    if args.pipeline:
        pipeline_dir = ROOT / "data" / "pipeline"
        if pipeline_dir.exists():
            for f in pipeline_dir.iterdir():
                rel = f.relative_to(ROOT)
                if args.dry_run:
                    print(f"[dry-run] would remove: {rel}")
                else:
                    try:
                        import shutil
                        if f.is_file():
                            f.unlink()
                        else:
                            shutil.rmtree(f)
                        removed.append(str(rel))
                    except Exception as e:
                        print(f"Skip {rel}: {e}")

    if args.processed:
        processed_dir = ROOT / "data" / "processed"
        if processed_dir.exists():
            for f in processed_dir.iterdir():
                rel = f.relative_to(ROOT)
                if args.dry_run:
                    print(f"[dry-run] would remove: {rel}")
                else:
                    try:
                        import shutil
                        if f.is_file():
                            f.unlink()
                        else:
                            shutil.rmtree(f)
                        removed.append(str(rel))
                    except Exception as e:
                        print(f"Skip {rel}: {e}")

    if not args.dry_run and removed:
        print(f"Removed {len(removed)} item(s). See STORAGE.md to regenerate.")
    elif args.dry_run:
        print("Run without --dry-run to apply.")


if __name__ == "__main__":
    main()
