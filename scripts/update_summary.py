#!/usr/bin/env python
"""Generate summary.md from data/MMR-Bench.csv and VLMEvalKit status files."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class Benchmark:
    name: str
    prefix: str
    short: str


BENCHMARKS = [
    Benchmark("MMStar", "MMStar", "MMStar"),
    Benchmark("RealWorldQA", "RealWorldQA", "RealWorldQA"),
    Benchmark("SEEDBench2_Plus", "SEEDBench2_Plus", "SEEDBench2+"),
    Benchmark("OCRBench", "OCRBench", "OCRBench"),
    Benchmark("MathVista_MINI", "MathVista", "MathVista"),
    Benchmark("MathVision", "MathVision", "MathVision"),
    Benchmark("MathVerse_MINI_Vision_Only", "MathVerse", "MathVerse"),
]


FAMILY_ORDER = [
    "GPT",
    "Gemini",
    "Claude",
    "Qwen3-VL",
    "Qwen2.5-VL",
    "InternVL3.5",
    "InternVL3",
    "Gemma",
    "GLM",
    "Ovis2.6",
    "Ovis2.5",
    "Ovis2",
    "LLaVA",
    "Other",
]


def model_family(model: str) -> str:
    model_lower = model.lower()
    if model.startswith("Qwen3-VL"):
        return "Qwen3-VL"
    if model.startswith("Qwen2.5-VL"):
        return "Qwen2.5-VL"
    if model.startswith("InternVL3_5") or model.startswith("InternVL3.5"):
        return "InternVL3.5"
    if model.startswith("InternVL3"):
        return "InternVL3"
    if model.startswith("Gemini"):
        return "Gemini"
    if model.startswith("Gemma"):
        return "Gemma"
    if model.startswith("GLM"):
        return "GLM"
    if model.startswith("Ovis2.6"):
        return "Ovis2.6"
    if model.startswith("Ovis2.5"):
        return "Ovis2.5"
    if model.startswith("Ovis2"):
        return "Ovis2"
    if model.startswith("Claude") or model_lower.startswith("claude"):
        return "Claude"
    if model_lower.startswith("gpt"):
        return "GPT"
    if model.startswith("LLaVA"):
        return "LLaVA"
    return "Other"


def family_rank(family: str) -> int:
    try:
        return FAMILY_ORDER.index(family)
    except ValueError:
        return len(FAMILY_ORDER)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/MMR-Bench.csv")
    parser.add_argument("--outputs", default="third_party/VLMEvalKit/outputs")
    parser.add_argument("--out", default="summary.md")
    return parser.parse_args()


def model_names(df: pd.DataFrame) -> list[str]:
    return [col[: -len("_prediction")] for col in df.columns if col.endswith("_prediction")]


def benchmark_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    idx = df["dataset_idx"].astype(str)
    return {bench.name: idx.str.startswith(f"{bench.prefix}_") for bench in BENCHMARKS}


def parse_bool(value) -> bool | None:
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "t", "yes", "y", "1"}:
        return True
    if text in {"false", "f", "no", "n", "0"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def bool_mean(series: pd.Series) -> float | None:
    non_null = series.dropna()
    if non_null.empty:
        return None
    parsed = non_null.map(parse_bool)
    return float(parsed.astype(bool).mean() * 100.0)


def fmt_score(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.1f}"


def collect_full_runs(outputs: Path) -> dict[str, dict[str, str]]:
    best: dict[str, dict[str, str]] = {}
    if not outputs.exists():
        return best

    grouped_statuses: dict[str, list[tuple[Path, dict]]] = {}
    for status_path in outputs.glob("**/status.json"):
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        datasets = status.get("datasets") or {}
        model = status.get("model_name")
        if not model:
            parent = status_path.parent
            model = parent.parent.name if parent.name.startswith("T") else parent.name
        grouped_statuses.setdefault(model, []).append((status_path, status))

        if len(datasets) < len(BENCHMARKS):
            continue
        if any(not isinstance(item, dict) or item.get("status") != "done" for item in datasets.values()):
            continue

        updated = max(
            [item.get("updated_at", "") for item in datasets.values() if isinstance(item, dict)]
            + [status.get("updated_at", "")]
        )
        current = best.get(model)
        if current is None or updated > current["updated_at"]:
            best[model] = {
                "updated_at": updated,
                "status_path": str(status_path),
            }

    required = {bench.name for bench in BENCHMARKS}
    for model, statuses in grouped_statuses.items():
        latest_by_dataset: dict[str, tuple[str, Path, dict]] = {}
        for status_path, status in statuses:
            datasets = status.get("datasets") or {}
            for dataset_name, item in datasets.items():
                if dataset_name not in required or not isinstance(item, dict):
                    continue
                updated = item.get("updated_at", "") or status.get("updated_at", "")
                current = latest_by_dataset.get(dataset_name)
                if current is None or updated > current[0]:
                    latest_by_dataset[dataset_name] = (updated, status_path, item)
        if required - set(latest_by_dataset):
            continue
        if any(item.get("status") != "done" for _, _, item in latest_by_dataset.values()):
            continue
        updated = max(updated for updated, _, _ in latest_by_dataset.values())
        status_path = max(latest_by_dataset.values(), key=lambda entry: entry[0])[1]
        current = best.get(model)
        if current is None or updated > current["updated_at"]:
            best[model] = {
                "updated_at": updated,
                "status_path": str(status_path.parent.parent if status_path.parent.name.startswith("T") else status_path),
            }
    return best


def collect_latest_partial(outputs: Path, full_runs: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    latest: dict[str, dict[str, str]] = {}
    if not outputs.exists():
        return []

    for status_path in outputs.glob("**/status.json"):
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        datasets = status.get("datasets") or {}
        if not datasets:
            continue

        model = status.get("model_name")
        if not model:
            parent = status_path.parent
            model = parent.parent.name if parent.name.startswith("T") else parent.name
        if model in full_runs:
            continue

        done = sum(1 for item in datasets.values() if isinstance(item, dict) and item.get("status") == "done")
        total = len(datasets)
        updated = max(
            [item.get("updated_at", "") for item in datasets.values() if isinstance(item, dict)]
            + [status.get("updated_at", "")]
        )
        record = {
            "model": model,
            "done": str(done),
            "total": str(total),
            "updated_at": updated,
            "status_path": str(status_path),
        }
        current = latest.get(model)
        if current is None or updated > current["updated_at"]:
            latest[model] = record
    return sorted(latest.values(), key=lambda item: item["updated_at"], reverse=True)


def build_rows(df: pd.DataFrame, full_runs: dict[str, dict[str, str]]) -> list[dict[str, object]]:
    masks = benchmark_masks(df)
    rows: list[dict[str, object]] = []
    for model in model_names(df):
        pred_col = f"{model}_prediction"
        correct_col = f"{model}_correct"
        token_col = f"{model}_token"
        if correct_col not in df:
            continue

        scores: dict[str, float | None] = {}
        for bench in BENCHMARKS:
            mask = masks[bench.name]
            scores[bench.name] = bool_mean(df.loc[mask, correct_col])

        macro_values = [score for score in scores.values() if score is not None]
        macro = sum(macro_values) / len(macro_values) if macro_values else None
        overall = bool_mean(df[correct_col])
        avg_token = None
        max_token = None
        if token_col in df:
            token_values = pd.to_numeric(df[token_col], errors="coerce").dropna()
            if not token_values.empty:
                avg_token = float(token_values.mean())
                max_token = int(token_values.max())

        rows.append(
            {
                "model": model,
                "family": model_family(model),
                "source": "local_full_run" if model in full_runs else "csv_existing",
                "updated_at": full_runs.get(model, {}).get("updated_at", ""),
                "overall": overall,
                "macro": macro,
                "scores": scores,
                "avg_token": avg_token,
                "max_token": max_token,
                "pred_non_null": int(df[pred_col].notna().sum()) if pred_col in df else 0,
                "correct_non_null": int(df[correct_col].notna().sum()),
            }
        )
    return sorted(rows, key=lambda item: (item["macro"] is not None, item["macro"] or -1), reverse=True)


def family_summary_rows(rows: list[dict[str, object]]) -> list[list[str]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["family"]), []).append(row)

    table_rows: list[list[str]] = []
    for family, items in sorted(grouped.items(), key=lambda kv: family_rank(kv[0])):
        ordered = sorted(items, key=lambda item: float(item["macro"] or -1), reverse=True)
        best = ordered[0]
        local_count = sum(1 for item in ordered if item["source"] == "local_full_run")
        table_rows.append(
            [
                family,
                str(len(ordered)),
                str(local_count),
                str(best["model"]),
                fmt_score(best["overall"]),
                fmt_score(best["macro"]),
                "<br>".join(str(item["model"]) for item in ordered),
            ]
        )
    return table_rows


def family_leaderboard_rows(rows: list[dict[str, object]]) -> list[list[str]]:
    ordered = sorted(
        rows,
        key=lambda item: (
            family_rank(str(item["family"])),
            -(float(item["macro"]) if item["macro"] is not None else -1.0),
            str(item["model"]),
        ),
    )
    table_rows: list[list[str]] = []
    current_family = None
    rank = 0
    for row in ordered:
        family = str(row["family"])
        if family != current_family:
            current_family = family
            rank = 1
        else:
            rank += 1
        scores = row["scores"]
        assert isinstance(scores, dict)
        table_rows.append(
            [
                family,
                str(rank),
                str(row["model"]),
                str(row["source"]),
                fmt_score(row["overall"]),
                fmt_score(row["macro"]),
                *[fmt_score(scores[bench.name]) for bench in BENCHMARKS],
                "NA" if row["avg_token"] is None else f"{float(row['avg_token']):.1f}",
                "NA" if row["max_token"] is None else str(row["max_token"]),
            ]
        )
    return table_rows


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def render_summary(
    df: pd.DataFrame,
    rows: list[dict[str, object]],
    full_runs: dict[str, dict[str, str]],
    partial_runs: list[dict[str, str]],
) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    masks = benchmark_masks(df)
    bench_rows = [
        [bench.name, bench.prefix, str(int(masks[bench.name].sum()))]
        for bench in BENCHMARKS
    ]

    score_headers = [
        "Rank",
        "Family",
        "Model",
        "Source",
        "Overall",
        "Macro Avg",
        *[bench.short for bench in BENCHMARKS],
        "Avg Tok",
        "Max Tok",
    ]
    score_rows = []
    for rank, row in enumerate(rows, start=1):
        scores = row["scores"]
        assert isinstance(scores, dict)
        score_rows.append(
            [
                str(rank),
                str(row["family"]),
                str(row["model"]),
                str(row["source"]),
                fmt_score(row["overall"]),
                fmt_score(row["macro"]),
                *[fmt_score(scores[bench.name]) for bench in BENCHMARKS],
                "NA" if row["avg_token"] is None else f"{float(row['avg_token']):.1f}",
                "NA" if row["max_token"] is None else str(row["max_token"]),
            ]
        )

    family_headers = [
        "Family",
        "Family Rank",
        "Model",
        "Source",
        "Overall",
        "Macro Avg",
        *[bench.short for bench in BENCHMARKS],
        "Avg Tok",
        "Max Tok",
    ]

    local_rows = []
    for row in sorted(
        [item for item in rows if item["source"] == "local_full_run"],
        key=lambda item: str(item["updated_at"]),
        reverse=True,
    ):
        model = str(row["model"])
        local_rows.append(
            [
                model,
                str(row["updated_at"]),
                fmt_score(row["overall"]),
                fmt_score(row["macro"]),
                full_runs.get(model, {}).get("status_path", ""),
            ]
        )

    existing_rows = [
        [str(row["model"]), fmt_score(row["overall"]), fmt_score(row["macro"])]
        for row in rows
        if row["source"] == "csv_existing"
    ]

    partial_rows = [
        [
            item["model"],
            f"{item['done']}/{item['total']}",
            item["updated_at"],
            item["status_path"],
        ]
        for item in partial_runs
    ]

    parts = [
        "# MMR-Bench Summary",
        "",
        f"Generated: {now}",
        "",
        "Source CSV: `data/MMR-Bench.csv`",
        "",
        "Scores are percentages computed from each model's `*_correct` column. "
        "`Overall` is row-weighted over all samples; `Macro Avg` is the unweighted average over the seven benchmark scores. "
        "`OCRBench` is shown as normalized accuracy percentage, equivalent to final score / 1000 * 100.",
        "",
        "## Benchmark Sizes",
        "",
        md_table(["Benchmark", "CSV prefix", "Samples"], bench_rows),
        "",
        "## Leaderboard",
        "",
        md_table(score_headers, score_rows),
        "",
        "## Model Families",
        "",
        "Families are inferred from model names. `Local Runs` counts models with a local seven-benchmark VLMEvalKit `status.json`.",
        "",
        md_table(
            ["Family", "Models", "Local Runs", "Best Model", "Best Overall", "Best Macro Avg", "All Models"],
            family_summary_rows(rows),
        ),
        "",
        "## Leaderboard By Family",
        "",
        "Rows are grouped by family, then sorted by `Macro Avg` within each family.",
        "",
        md_table(family_headers, family_leaderboard_rows(rows)),
        "",
        "## Local Full VLMEvalKit Runs",
        "",
        "These models have a local `status.json` with all seven benchmark statuses marked `done`.",
        "",
        md_table(["Model", "Completed at", "Overall", "Macro Avg", "Status file"], local_rows)
        if local_rows
        else "None.",
        "",
        "## CSV-Existing Complete Models",
        "",
        "These models are complete in the CSV, but no matching local seven-benchmark `status.json` was found under `third_party/VLMEvalKit/outputs`.",
        "",
        md_table(["Model", "Overall", "Macro Avg"], existing_rows) if existing_rows else "None.",
        "",
        "## Partial Local Outputs Not Counted As Full Runs",
        "",
        md_table(["Model", "Done", "Updated at", "Status file"], partial_rows) if partial_rows else "None.",
        "",
        "## Update Command",
        "",
        "```bash",
        "python scripts/update_summary.py",
        "```",
        "",
    ]
    return "\n".join(parts)


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    outputs = Path(args.outputs)
    out_path = Path(args.out)

    df = pd.read_csv(csv_path, low_memory=False)
    if "dataset_idx" not in df.columns:
        raise ValueError(f"{csv_path} has no dataset_idx column")

    full_runs = collect_full_runs(outputs)
    partial_runs = collect_latest_partial(outputs, full_runs)
    rows = build_rows(df, full_runs)
    out_path.write_text(render_summary(df, rows, full_runs, partial_runs), encoding="utf-8")
    print(f"wrote={out_path}")
    print(f"models={len(rows)}")
    print(f"local_full_runs={sum(1 for row in rows if row['source'] == 'local_full_run')}")


if __name__ == "__main__":
    main()
