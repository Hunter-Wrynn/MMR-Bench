from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mmrbench.data import MMData
from mmrbench.embedding import Embedding


class RandomRouter:
    def __init__(self, args):
        self.args = args
        self.out_dir = Path("./outputs")
        self.out_dir.mkdir(exist_ok=True, parents=True)
        self.embeder = Embedding(args)  # interface compatibility
        self.processor = None
        self.model_ids = None
        self._name = "random"
        self.test_df = None

    def fit(self):
        self.processor = MMData(self.args)
        self.test_df = self.processor.test_data
        self.model_ids = list(self.processor.model_ids)

    def evaluate_under_cost(self, n_trials: int = 50):
        if self.processor is None:
            self.fit()

        rng = np.random.default_rng(int(getattr(self.args, "random_state", 42)))
        test_df = self.test_df
        n_samples = test_df.shape[0]
        n_models = len(self.model_ids)

        points = []
        for _ in range(max(1, n_trials)):
            chosen_idx = rng.integers(0, n_models, size=n_samples)
            chosen_ids = [self.model_ids[i] for i in chosen_idx]
            perf_cols = [self.processor.model_list[mid] + "_is_correct" for mid in chosen_ids]
            cost_cols = [self.processor.model_list[mid] + "_cost" for mid in chosen_ids]
            row_idx = np.arange(n_samples)
            col_perf = [test_df.columns.get_loc(col) for col in perf_cols]
            col_cost = [test_df.columns.get_loc(col) for col in cost_cols]
            chosen_perf = test_df.to_numpy()[row_idx, col_perf]
            chosen_cost = test_df.to_numpy()[row_idx, col_cost]
            points.append({"cost": float(np.nanmean(chosen_cost)), "performance": float(np.nanmean(chosen_perf))})

        points.sort(key=lambda x: x["cost"])
        out_path = self.out_dir / f"{self._name}_{self.args.dataset}_{self.args.mode}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(points, ensure_ascii=False, indent=2), encoding="utf-8")
        return points

