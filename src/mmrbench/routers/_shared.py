from __future__ import annotations

import json
from pathlib import Path


def save_points(out_dir: Path, name: str, dataset: str, mode: str, points: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}_{dataset}_{mode}.json"
    path.write_text(json.dumps(points, ensure_ascii=False, indent=2), encoding="utf-8")

