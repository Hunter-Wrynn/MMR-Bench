from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    csv_path: Path
    images_dir: Path | None


def _infer_model_names_from_columns(cols: Iterable[str]) -> list[str]:
    correct = set()
    cost = set()
    for c in cols:
        if c.endswith("_correct"):
            correct.add(c[: -len("_correct")])
        if c.endswith("_cost"):
            cost.add(c[: -len("_cost")])
    model_names = sorted(correct & cost)
    return model_names


def _default_spec(data_root: Path, name: str) -> DatasetSpec:
    return DatasetSpec(
        name=name,
        csv_path=data_root / f"{name}.csv",
        images_dir=(data_root / name) if (data_root / name).exists() else None,
    )


def _canon(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


_MMR_PREFIX_TO_DIR = {
    # dataset_idx prefix -> images folder name under data_root
    "MathVerse": "MathVerse",
    "MathVision": "MathVision",
    "MathVista": "MathVista",
    "OCRBench": "OCRBench",
    "RealWorldQA": "RealWorldQA",
    "MMStar": "MMStar",
    # Special case in your merged CSV: SEEDBench2_Plus_<id> -> SEEDBenchv2Plus/<id>.jpg
    "SEEDBench2_Plus": "SEEDBenchv2Plus",
}

_CANON_TO_MMR_PREFIX = {
    _canon("mathverse"): "MathVerse",
    _canon("mathvision"): "MathVision",
    _canon("mathvista"): "MathVista",
    _canon("ocrbench"): "OCRBench",
    _canon("realworldqa"): "RealWorldQA",
    _canon("mmstar"): "MMStar",
    _canon("seedbench"): "SEEDBench2_Plus",
    _canon("seedbench2plus"): "SEEDBench2_Plus",
    _canon("seedbenchv2plus"): "SEEDBench2_Plus",
    _canon("seedbench2_plus"): "SEEDBench2_Plus",
}


def _choose_prefix(dataset_idx: str) -> str | None:
    s = str(dataset_idx)
    best = None
    for p in _MMR_PREFIX_TO_DIR.keys():
        if s == p or s.startswith(p + "_"):
            if best is None or len(p) > len(best):
                best = p
    return best


def _img_path_from_dataset_idx(data_root: Path, dataset_idx: str) -> tuple[str, str]:
    """
    Returns (dataset_dir_name, img_path).
    """
    prefix = _choose_prefix(dataset_idx)
    if prefix is None:
        return ("unknown", "")
    folder = _MMR_PREFIX_TO_DIR[prefix]

    remainder = str(dataset_idx)
    if remainder.startswith(prefix + "_"):
        remainder = remainder[len(prefix) + 1 :]
    else:
        remainder = remainder.split("_")[-1]

    base = (data_root / folder).resolve()
    for ext in [".jpg", ".png", ".jpeg", ".webp"]:
        candidate = base / f"{remainder}{ext}"
        if candidate.exists():
            return (folder, str(candidate))
    return (folder, str((base / f"{remainder}.jpg").resolve()))


class MMData:
    """
    Load MMR-Bench-style *offline outcomes* for routing.

    Expected CSV schema per dataset:
      - question: str
      - answer: str (optional)
      - img_path: str (recommended, absolute or relative)
      - OR dataset_idx: str (optional; used only to construct img_path if img_path missing)
      - For each model name M: columns M_correct (0/1) and M_cost (float)

    Internally we convert to:
      - M_is_correct (float)
      - M_cost (float)
    """

    def __init__(self, args, force_split: bool = False):
        self.args = args
        self.data_root = Path(getattr(args, "data_root", Path("./data")))

        dataset_names = str(getattr(args, "dataset", "toy")).split("+")
        self.dataset_names = [d.strip() for d in dataset_names if d.strip()]

        merged_path = self.data_root / "MMR_Bench.csv"
        use_merged = merged_path.exists() and all(_canon(n) in _CANON_TO_MMR_PREFIX for n in self.dataset_names)

        if use_merged:
            all_data = self._load_merged_mmr_bench(merged_path)
        else:
            self.specs = [_default_spec(self.data_root, name.lower()) for name in self.dataset_names]
            all_frames = [self._load_one(spec) for spec in self.specs]
            if not all_frames:
                raise ValueError("No datasets loaded.")
            all_data = pd.concat(all_frames, ignore_index=True)

        self.model_list = _infer_model_names_from_columns(all_data.columns)
        if not self.model_list:
            raise ValueError(
                "Failed to infer model names from columns. Expected pairs like '<model>_correct' and '<model>_cost'."
            )
        self.model_ids = list(range(len(self.model_list)))

        # Ensure required columns exist
        if "question" not in all_data.columns:
            raise ValueError("CSV must contain a 'question' column.")

        # Build img_path (MMR_Bench.csv uses dataset_idx -> <dataset_dir>/<id>.jpg)
        if "img_path" not in all_data.columns or all_data["img_path"].isna().all():
            if "dataset_idx" in all_data.columns:
                ds_names = []
                paths = []
                for idx in all_data["dataset_idx"].tolist():
                    ds, p = _img_path_from_dataset_idx(self.data_root, idx)
                    ds_names.append(ds)
                    paths.append(p)
                # If dataset name not set (or merged), prefer inferred directory name
                if "__dataset_name" not in all_data.columns or (all_data["__dataset_name"].astype(str) == "").all():
                    all_data["__dataset_name"] = ds_names
                all_data["img_path"] = paths
            else:
                all_data["img_path"] = [""] * len(all_data)

        processed = self._to_internal_schema(all_data)

        if str(getattr(args, "split", "in")) == "in":
            self.train_data, self.test_data = train_test_split(
                processed,
                train_size=float(getattr(args, "train_percent", 0.2)),
                shuffle=True,
                random_state=int(getattr(args, "random_state", 42)),
            )
        else:
            # dataset-level split: first portion as train, rest as test
            out_p = float(getattr(args, "out_percent", 0.2))
            n_train = max(1, int(len(self.dataset_names) * out_p)) if self.dataset_names else 0
            train_names = set(self.dataset_names[:n_train])
            is_train = processed["__dataset_name"].isin(train_names)
            self.train_data = processed[is_train].copy()
            self.test_data = processed[~is_train].copy()

        # basic cost normalization (optional; used by some analyses)
        self.model_costs_norm = self._estimate_model_costs_norm(processed)

    def _load_merged_mmr_bench(self, merged_path: Path) -> pd.DataFrame:
        df = pd.read_csv(merged_path)
        if "dataset_idx" not in df.columns:
            raise ValueError("MMR_Bench.csv must contain a 'dataset_idx' column.")

        wanted_prefixes = {_CANON_TO_MMR_PREFIX[_canon(n)] for n in self.dataset_names}
        keep_mask = []
        ds_names = []
        for idx in df["dataset_idx"].tolist():
            prefix = _choose_prefix(idx)
            keep = prefix in wanted_prefixes
            keep_mask.append(keep)
            ds_names.append(_MMR_PREFIX_TO_DIR.get(prefix, "unknown") if prefix else "unknown")

        df = df[np.array(keep_mask, dtype=bool)].copy()
        df["__dataset_name"] = [d for k, d in zip(keep_mask, ds_names, strict=False) if k]
        return df

    def _load_one(self, spec: DatasetSpec) -> pd.DataFrame:
        if not spec.csv_path.exists():
            raise FileNotFoundError(f"Dataset CSV not found: {spec.csv_path}")
        df = pd.read_csv(spec.csv_path)
        df["__dataset_name"] = spec.name

        # If img_path is relative, resolve against images_dir or data_root
        if "img_path" in df.columns:
            df["img_path"] = df["img_path"].fillna("").astype(str)
            resolved = []
            for p in df["img_path"].tolist():
                if not p:
                    resolved.append("")
                    continue
                path = Path(p)
                if path.is_absolute():
                    resolved.append(str(path))
                else:
                    base = spec.images_dir or self.data_root
                    resolved.append(str((base / path).resolve()))
            df["img_path"] = resolved

        return df

    def _construct_img_paths(self, df: pd.DataFrame) -> list[str]:
        if "dataset_idx" not in df.columns:
            return [""] * len(df)

        # Heuristic: take the last underscore token as image id; for seedbench use the 3rd token (legacy).
        def img_id_from_dataset_idx(dataset_name: str, dataset_idx: str) -> str:
            parts = str(dataset_idx).split("_")
            if dataset_name.lower().startswith("seedbench") and len(parts) >= 3:
                return parts[2]
            return parts[-1] if parts else str(dataset_idx)

        img_paths: list[str] = []
        for dataset_name, dataset_idx in zip(df["__dataset_name"].tolist(), df["dataset_idx"].tolist(), strict=False):
            spec = _default_spec(self.data_root, str(dataset_name))
            if spec.images_dir is None:
                img_paths.append("")
                continue
            img_id = img_id_from_dataset_idx(str(dataset_name), str(dataset_idx))
            img_paths.append(str((spec.images_dir / f"{img_id}.jpg").resolve()))
        return img_paths

    def _to_internal_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame()
        out["__dataset_name"] = df["__dataset_name"].astype(str)
        out["question"] = df["question"].astype(str)
        if "answer" in df.columns:
            out["answer"] = df["answer"].astype(str)
        else:
            out["answer"] = ""
        out["img_path"] = df["img_path"].fillna("").astype(str)

        for model_name in self.model_list:
            correct_col = f"{model_name}_correct"
            cost_col = f"{model_name}_cost"
            out[f"{model_name}_is_correct"] = df[correct_col].astype(float) if correct_col in df.columns else np.nan
            out[f"{model_name}_cost"] = df[cost_col].astype(float) if cost_col in df.columns else np.nan

        # Fill missing numeric entries with column means (nan-safe)
        for col in out.columns:
            if col in {"__dataset_name", "question", "answer", "img_path"}:
                continue
            if out[col].dtype.kind not in {"f", "i"}:
                continue
            if out[col].isna().any():
                non_null = out[col].dropna()
                if len(non_null) > 0:
                    out[col] = out[col].fillna(float(non_null.mean()))
        return out

    def _estimate_model_costs_norm(self, df: pd.DataFrame) -> dict[int, float]:
        # Lightweight estimate: mean per-model cost over all rows, normalized by max.
        means = []
        for model_name in self.model_list:
            means.append(float(np.nanmean(df[f"{model_name}_cost"].to_numpy(dtype=float))))
        max_cost = max(means) if means else 1.0
        if max_cost <= 0:
            max_cost = 1.0
        return {i: (means[i] / max_cost) for i in range(len(means))}
