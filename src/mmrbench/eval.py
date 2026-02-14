from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _pareto_frontier(points: list[dict[str, float]]) -> list[dict[str, float]]:
    pts = [p for p in points if p.get("cost") is not None and p.get("performance") is not None]
    pts.sort(key=lambda x: (x["cost"], -x["performance"]))
    out = []
    best = -math.inf
    for p in pts:
        if p["performance"] > best:
            out.append(p)
            best = p["performance"]
    return out


def _trapz_auc(xs: np.ndarray, ys: np.ndarray) -> float:
    if len(xs) < 2:
        return float("nan")
    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]
    return float(np.trapz(ys, xs))


@dataclass(frozen=True)
class SummaryMetrics:
    nAUC: float
    Ps: float
    QNC: float


class RouterEvaluator:
    def __init__(self, router):
        self.router = router
        self.processor = getattr(router, "processor", None)

    def _get_single_model_baselines(self) -> list[dict[str, float]]:
        test_df = self.router.processor.test_data
        results = []
        for mid in self.router.model_ids:
            model_name = self.router.processor.model_list[mid]
            acc = float(np.nanmean(test_df[f"{model_name}_is_correct"].to_numpy(dtype=float)))
            cost = float(np.nanmean(test_df[f"{model_name}_cost"].to_numpy(dtype=float)))
            results.append({"model_id": int(mid), "model_name": model_name, "accuracy": acc, "avg_cost": cost})
        return results

    def eval_all(self) -> dict[str, Any]:
        # Router-specific curve generation (kept compatible with existing baselines)
        points = self.router.evaluate_under_cost()
        pareto = _pareto_frontier(points)

        xs = np.array([p["cost"] for p in pareto], dtype=float)
        ys = np.array([p["performance"] for p in pareto], dtype=float)
        cmin = float(np.nanmin(xs)) if len(xs) else float("nan")
        cmax = float(np.nanmax(xs)) if len(xs) else float("nan")

        auc = _trapz_auc(xs, ys)
        nAUC = float(auc / (cmax - cmin)) if (not math.isnan(auc) and cmax > cmin) else float("nan")
        Ps = float(np.nanmax(ys)) if len(ys) else float("nan")

        baselines = self._get_single_model_baselines()
        best_model = max(baselines, key=lambda x: x["accuracy"])
        pbest = best_model["accuracy"]
        cbest = best_model["avg_cost"]

        # QNC: minimum cost on pareto achieving pbest, normalized by cbest
        qnc = float("inf")
        for p in pareto:
            if p["performance"] >= pbest:
                qnc = float(p["cost"] / cbest) if cbest > 0 else float("inf")
                break

        # Save curve if router chose to write it; also return summary.
        return {
            "router": getattr(self.router, "_name", type(self.router).__name__),
            "dataset": getattr(self.router.args, "dataset", ""),
            "mode": getattr(self.router.args, "mode", ""),
            "metrics": {"nAUC": nAUC, "Ps": Ps, "QNC": qnc},
            "single_model": baselines,
            "pareto": pareto,
        }

