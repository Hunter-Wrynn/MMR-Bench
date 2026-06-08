from __future__ import annotations

import argparse
import shutil
import tarfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download and place MMR-Bench data under ./data")
    p.add_argument("--repo", type=str, default="gh0stHunter/MMR-Bench", help="Hugging Face dataset repo id")
    p.add_argument("--revision", type=str, default=None, help="HF revision (branch/tag/commit)")
    p.add_argument("--dest", type=Path, default=Path("data"), help="Destination directory (contains MMR_Bench.csv)")
    p.add_argument("--force", action="store_true", help="Overwrite existing files")
    p.add_argument("--no-extract-images", action="store_true", help="Keep images.tar.gz without extracting it")
    return p.parse_args()


def _safe_extract_tar(archive: Path, dest: Path) -> None:
    dest_resolved = dest.resolve()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            target = (dest / member.name).resolve()
            if not target.is_relative_to(dest_resolved):
                raise SystemExit(f"Unsafe tar member path: {member.name}")
        tar.extractall(dest)


def _replace_path(src: Path, dst: Path, force: bool) -> None:
    if dst.exists() or dst.is_symlink():
        if not force:
            return
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


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
        # Current HF repository layout.
        "MMR-Bench.csv",
        "images.tar.gz",
        # Legacy/local expanded layout supported by the loader.
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

    csv_src = None
    for name in ["MMR_Bench.csv", "MMR-Bench.csv"]:
        candidate = local_dir / name
        if candidate.exists():
            csv_src = candidate
            break
    if csv_src is None:
        raise SystemExit("Expected MMR-Bench CSV missing in downloaded snapshot.")

    _replace_path(csv_src, dest / "MMR_Bench.csv", force=True)

    # Copy expanded image folders when present.
    for rel in [
        Path("MathVerse"),
        Path("MathVision"),
        Path("MathVista"),
        Path("OCRBench"),
        Path("RealWorldQA"),
        Path("MMStar"),
        Path("SEEDBenchv2Plus"),
    ]:
        src = local_dir / rel
        if src.exists():
            _replace_path(src, dest / rel, force=args.force)

    archive = local_dir / "images.tar.gz"
    if archive.exists():
        archive_dst = dest / "images.tar.gz"
        _replace_path(archive, archive_dst, force=args.force)
        if not args.no_extract_images:
            _safe_extract_tar(archive_dst, dest)

    print(f"Done. Data is ready under: {dest}")


if __name__ == "__main__":
    main()
