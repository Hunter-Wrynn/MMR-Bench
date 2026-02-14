from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download and place MMR-Bench data under ./data")
    p.add_argument("--repo", type=str, default="gh0stHunter/MMR-Bench", help="Hugging Face dataset repo id")
    p.add_argument("--revision", type=str, default=None, help="HF revision (branch/tag/commit)")
    p.add_argument("--dest", type=Path, default=Path("data"), help="Destination directory (contains MMR_Bench.csv)")
    p.add_argument("--force", action="store_true", help="Overwrite existing files")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dest: Path = args.dest
    dest.mkdir(parents=True, exist_ok=True)

    mmr_csv = dest / "MMR_Bench.csv"
    if mmr_csv.exists() and not args.force:
        raise SystemExit(f"{mmr_csv} already exists. Use --force to overwrite.")

    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        raise SystemExit("Missing dependency: huggingface-hub. Install with: pip install -e '.[hf]'") from e

    allow_patterns = [
        "MMR_Bench.csv",
        "MathVerse/*",
        "MathVision/*",
        "MathVista/*",
        "OCRBench/*",
        "RealWorldQA/*",
        "MMStar/*",
        "SEEDBenchv2Plus/*",
    ]

    cache_dir = dest / "_hf_snapshot"
    if cache_dir.exists() and args.force:
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    local_dir = snapshot_download(
        repo_id=args.repo,
        repo_type="dataset",
        revision=args.revision,
        allow_patterns=allow_patterns,
        local_dir=str(cache_dir),
        local_dir_use_symlinks=False,
    )

    local_dir = Path(local_dir)

    # Copy into dest (flatten so files/folders land directly under dest)
    for rel in [
        Path("MMR_Bench.csv"),
        Path("MathVerse"),
        Path("MathVision"),
        Path("MathVista"),
        Path("OCRBench"),
        Path("RealWorldQA"),
        Path("MMStar"),
        Path("SEEDBenchv2Plus"),
    ]:
        src = local_dir / rel
        if not src.exists():
            raise SystemExit(f"Expected path missing in downloaded snapshot: {src}")
        dst = dest / rel
        if dst.exists() and args.force:
            if dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

    print(f"Done. Data is ready under: {dest}")


if __name__ == "__main__":
    main()

