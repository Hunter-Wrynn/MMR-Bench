#!/usr/bin/env python
"""Merge VLMEvalKit predictions and per-sample scores into MMR-Bench.csv."""

from __future__ import annotations

import argparse
import ast
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd
from transformers import AutoTokenizer


@dataclass(frozen=True)
class DatasetSpec:
    vlmeval_name: str
    mmr_prefix: str
    kind: str


SPECS = [
    DatasetSpec("MMStar", "MMStar", "mcq"),
    DatasetSpec("RealWorldQA", "RealWorldQA", "mcq"),
    DatasetSpec("SEEDBench2_Plus", "SEEDBench2_Plus", "mcq"),
    DatasetSpec("OCRBench", "OCRBench", "ocr"),
    DatasetSpec("MathVista_MINI", "MathVista", "mathvista"),
    DatasetSpec("MathVision", "MathVision", "mathvision"),
    DatasetSpec("MathVerse_MINI_Vision_Only", "MathVerse", "mathverse"),
]


def load_tokenizer(model_path: str, tokenizer_kwargs: dict):
    tokenizer_config = Path(model_path) / "tokenizer_config.json"
    if tokenizer_config.exists():
        try:
            tokenizer_class = json.loads(tokenizer_config.read_text(encoding="utf-8")).get("tokenizer_class")
        except Exception:
            tokenizer_class = None
        if tokenizer_class == "Qwen2Tokenizer":
            from transformers import Qwen2Tokenizer

            return Qwen2Tokenizer.from_pretrained(model_path, **tokenizer_kwargs)
    return AutoTokenizer.from_pretrained(model_path, **tokenizer_kwargs)


def normalize_index(value) -> str:
    if pd.isna(value):
        raise ValueError("empty index")
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def find_latest(model_root: Path, filename: str) -> Path:
    matches = sorted(
        model_root.glob(f"T*/{filename}"),
        key=lambda p: (p.stat().st_mtime, str(p)),
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError(f"Missing VLMEvalKit file: {model_root}/T*/{filename}")
    return matches[0]


def read_xlsx(path: Path) -> pd.DataFrame:
    return pd.read_excel(path)


def ocr_correct(row: pd.Series) -> bool:
    prediction = str(row["prediction"])
    try:
        answers = ast.literal_eval(str(row["answer"]))
    except Exception as exc:
        raise ValueError(f"Cannot parse OCRBench answer for index={row.get('index')}: {row['answer']}") from exc
    category = row["category"]
    if category == "Handwritten Mathematical Expression Recognition":
        pred = prediction.strip().replace("\n", " ").replace(" ", "")
        return any(str(ans).strip().replace("\n", " ").replace(" ", "") in pred for ans in answers)
    pred = prediction.lower().strip().replace("\n", " ")
    return any(str(ans).lower().strip().replace("\n", " ") in pred for ans in answers)


def load_correct_map(
    model_root: Path,
    model: str,
    judge: str,
    spec: DatasetSpec,
) -> dict[str, bool]:
    base = f"{model}_{spec.vlmeval_name}"
    if spec.kind == "mcq":
        result = read_xlsx(find_latest(model_root, f"{base}_{judge}_result.xlsx"))
        return {
            f"{spec.mmr_prefix}_{normalize_index(row['index'])}": bool(row["hit"])
            for _, row in result.iterrows()
        }
    if spec.kind == "ocr":
        pred = read_xlsx(find_latest(model_root, f"{base}.xlsx"))
        return {
            f"{spec.mmr_prefix}_{normalize_index(row['index'])}": ocr_correct(row)
            for _, row in pred.iterrows()
        }
    if spec.kind == "mathvista":
        from vlmeval.dataset.utils.mathvista import post_check

        result = read_xlsx(find_latest(model_root, f"{base}_{judge}.xlsx"))
        return {
            f"{spec.mmr_prefix}_{normalize_index(row['index'])}": bool(post_check(row, prefetch=False))
            for _, row in result.iterrows()
        }
    if spec.kind == "mathvision":
        from vlmeval.dataset.utils.mathv import post_check

        result = read_xlsx(find_latest(model_root, f"{base}_{judge}.xlsx"))
        return {
            f"{spec.mmr_prefix}_{normalize_index(row['index'])}": bool(post_check(row, prefetch=False))
            for _, row in result.iterrows()
        }
    if spec.kind == "mathverse":
        result = read_xlsx(find_latest(model_root, f"{base}_{judge}_score.xlsx"))
        return {
            f"{spec.mmr_prefix}_{normalize_index(row['index'])}": bool(row["score"])
            for _, row in result.iterrows()
        }
    raise ValueError(f"Unsupported dataset kind: {spec.kind}")


def load_prediction_map(model_root: Path, model: str, spec: DatasetSpec) -> dict[str, str]:
    pred = read_xlsx(find_latest(model_root, f"{model}_{spec.vlmeval_name}.xlsx"))
    if "prediction" not in pred.columns:
        raise ValueError(f"{spec.vlmeval_name} prediction file has no `prediction` column")
    return {
        f"{spec.mmr_prefix}_{normalize_index(row['index'])}": "" if pd.isna(row["prediction"]) else str(row["prediction"])
        for _, row in pred.iterrows()
    }


def csv_prediction_text(text: str) -> str:
    if not text.strip():
        return "[EMPTY_RESPONSE]"
    return text


def insert_columns(df: pd.DataFrame, values: dict[str, pd.Series]) -> pd.DataFrame:
    for col in values:
        if col in df.columns:
            df = df.drop(columns=[col])

    cost_start = next((i for i, col in enumerate(df.columns) if col.endswith("_cost")), len(df.columns))
    ordered = list(df.columns[:cost_start])
    ordered.extend([col for col in values if not col.endswith("_cost")])
    ordered.extend(df.columns[cost_start:])
    ordered.extend([col for col in values if col.endswith("_cost")])

    for col, series in values.items():
        df[col] = series
    return df[ordered]


def build_model_columns(
    df: pd.DataFrame,
    model_root: Path,
    model: str,
    judge: str,
    tokenizer: AutoTokenizer,
) -> tuple[dict[str, pd.Series], dict[str, int]]:
    predictions: dict[str, str] = {}
    correct: dict[str, bool] = {}
    source_counts: dict[str, int] = {}

    for spec in SPECS:
        pred_map = load_prediction_map(model_root, model, spec)
        correct_map = load_correct_map(model_root, model, judge, spec)
        missing_scores = sorted(set(pred_map) - set(correct_map))
        if missing_scores:
            raise ValueError(f"{spec.vlmeval_name}: {len(missing_scores)} predictions have no score, e.g. {missing_scores[:5]}")
        predictions.update(pred_map)
        correct.update(correct_map)
        source_counts[spec.vlmeval_name] = len(pred_map)

    target_ids = set(df["dataset_idx"].astype(str))
    missing_predictions = sorted(target_ids - set(predictions))
    extra_predictions = sorted(set(predictions) - target_ids)
    missing_correct = sorted(target_ids - set(correct))
    if missing_predictions or missing_correct or extra_predictions:
        raise ValueError(
            "VLMEvalKit/MMR-Bench index mismatch: "
            f"missing_predictions={len(missing_predictions)} {missing_predictions[:8]}, "
            f"missing_correct={len(missing_correct)} {missing_correct[:8]}, "
            f"extra_predictions={len(extra_predictions)} {extra_predictions[:8]}"
        )

    raw_pred_series = df["dataset_idx"].map(predictions).astype(str)
    pred_series = raw_pred_series.map(csv_prediction_text)
    correct_series = df["dataset_idx"].map(correct).astype(bool)
    token_series = raw_pred_series.map(lambda text: len(tokenizer.encode(text, add_special_tokens=False))).astype(int)
    cost_series = pd.Series([0.0] * len(df), index=df.index)

    cols = {
        f"{model}_prediction": pred_series,
        f"{model}_correct": correct_series,
        f"{model}_token": token_series,
        f"{model}_cost": cost_series,
    }
    return cols, source_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/MMR-Bench.csv")
    parser.add_argument("--run-root", required=True, help="VLMEvalKit work-dir root")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--judge", default="Qwen3.5-122B-A10B")
    parser.add_argument("--fix-mistral-regex", action="store_true")
    parser.add_argument("--backup", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    vlmeval_path = repo_root / "third_party" / "VLMEvalKit"
    sys.path.insert(0, str(vlmeval_path))

    csv_path = Path(args.csv)
    model_root = Path(args.run_root) / args.model
    if not model_root.exists():
        raise FileNotFoundError(model_root)

    df = pd.read_csv(csv_path)
    if "dataset_idx" not in df.columns:
        raise ValueError(f"{csv_path} has no dataset_idx column")

    tokenizer_kwargs = {"trust_remote_code": True}
    if args.fix_mistral_regex:
        tokenizer_kwargs["fix_mistral_regex"] = True
    tokenizer = load_tokenizer(args.model_path, tokenizer_kwargs)
    cols, source_counts = build_model_columns(df, model_root, args.model, args.judge, tokenizer)
    merged = insert_columns(df, cols)

    if args.backup:
        backup = csv_path.with_suffix(f".before_{args.model}.csv")
        shutil.copy2(csv_path, backup)
        print(f"backup={backup}")

    merged.to_csv(csv_path, index=False)
    print(f"merged={csv_path}")
    print(f"rows={len(merged)}")
    print("source_counts=" + ",".join(f"{k}:{v}" for k, v in source_counts.items()))
    for col in cols:
        print(f"{col}: non_null={merged[col].notna().sum()}")


if __name__ == "__main__":
    main()
