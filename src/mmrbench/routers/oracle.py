from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mmrbench.data import MMData
from mmrbench.embedding import Embedding


class OracleRouter:
    def __init__(self, args):
        self.args = args
        self.out_dir = Path("./outputs")
        self.out_dir.mkdir(exist_ok=True, parents=True)
        self.embeder = Embedding(args)
        self.processor = None
        self.model_ids = None
        self._name = "oracle"
        self.test_df = None
        self.true_perf = None
        self.true_cost = None

    def fit(self):
        self.processor = MMData(self.args)
        self.model_ids = list(self.processor.model_ids)
        self.test_df = self.processor.test_data.copy()
        perf_cols = [f"{m}_is_correct" for m in self.processor.model_list]
        cost_cols = [f"{m}_cost" for m in self.processor.model_list]
        self.true_perf = self.test_df[perf_cols].to_numpy(dtype=float)
        self.true_cost = self.test_df[cost_cols].to_numpy(dtype=float)

    def evaluate_under_cost(self):
        if self.true_perf is None:
            self.fit()

        perf = self.true_perf
        cost = self.true_cost
        n, _ = perf.shape
        all_costs = np.sort(np.unique(cost))
        points = []
        for C in all_costs:
            mask = cost <= C
            masked_perf = np.where(mask, perf, -np.inf)
            best_idx = np.argmax(masked_perf, axis=1)
            selected_costs = cost[np.arange(n), best_idx]
            selected_perf = perf[np.arange(n), best_idx]
            points.append({"cost": float(np.mean(selected_costs)), "performance": float(np.mean(selected_perf))})

        out_path = self.out_dir / f"{self._name}_{self.args.dataset}_{self.args.mode}.json"
        out_path.write_text(json.dumps(points, ensure_ascii=False, indent=2), encoding="utf-8")
        return points

