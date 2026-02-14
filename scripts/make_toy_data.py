from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


def main() -> None:
    out_root = Path("data/toy").resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    img_dir = out_root / "toy"
    img_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(42)
    np_rng = np.random.default_rng(42)

    models = [
        "Qwen2.5-VL-3B-Instruct",
        "Qwen2.5-VL-7B-Instruct",
        "InternVL3-78B",
        "gpt-5-2025-08-07",
    ]

    rows = []
    for i in range(200):
        # Make a simple image with a digit so image modality is non-trivial.
        digit = rng.randint(0, 9)
        img = Image.new("RGB", (224, 224), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((90, 90), str(digit), fill=(0, 0, 0))
        img_path = img_dir / f"{i}.jpg"
        img.save(img_path)

        q = f"What digit is shown? Answer with a single digit."

        # Simulate model outcomes: stronger models are more accurate and more expensive.
        base_cost = [0.03, 0.07, 0.70, 10.0]
        base_acc = [0.65, 0.75, 0.85, 0.93]
        row = {"question": q, "answer": str(digit), "img_path": str(img_path), "dataset_idx": f"toy_{i}"}
        difficulty = float(np_rng.uniform(0.0, 1.0))
        for m, c, a in zip(models, base_cost, base_acc, strict=True):
            p = max(0.05, min(0.99, a - 0.25 * difficulty))
            row[f"{m}_correct"] = 1.0 if np_rng.uniform() < p else 0.0
            row[f"{m}_cost"] = float(c * (0.8 + 0.4 * np_rng.uniform()))
        rows.append(row)

    df = pd.DataFrame(rows)
    (out_root / "toy.csv").write_text(df.to_csv(index=False), encoding="utf-8")
    print(f"Wrote toy dataset to {out_root}")


if __name__ == "__main__":
    main()

