from __future__ import annotations

import numpy as np
from pathlib import Path
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor

from mmrbench.data import MMData
from mmrbench.embedding import Embedding
from mmrbench.routers._shared import save_points


class LinearMFRouter:
    def __init__(self, args):
        self.args = args
        self.out_dir = Path("./outputs")
        self.out_dir.mkdir(exist_ok=True, parents=True)
        self._name = "linearmf"
        self.embeder = Embedding(args)
        self.processor = None
        self.model_ids = None
        self.svd = None
        self.reg_perf = None
        self.reg_cost = None
        self.test_Z = None

    def _encode_features(self, texts, img_paths=None, mode="train"):
        mode_flag = self.args.mode[0] if mode == "train" else self.args.mode[1]
        if mode_flag == "1":
            return self.embeder.run_text(texts).astype(np.float32)
        if mode_flag == "3":
            return self.embeder.run_img(img_paths).astype(np.float32)
        X_text = self.embeder.run_text(texts)
        X_img = np.zeros_like(X_text) if img_paths is None else self.embeder.run_img(img_paths)
        if X_text.shape[1] != X_img.shape[1]:
            d = min(X_text.shape[1], X_img.shape[1])
            X_text = X_text[:, :d]
            X_img = X_img[:, :d]
        return ((X_text + X_img) / 2.0).astype(np.float32)

    def fit(self):
        self.processor = MMData(self.args)
        train_df = self.processor.train_data
        self.model_ids = list(self.processor.model_ids)

        texts = train_df["question"].astype(str).tolist()
        img_paths = train_df["img_path"].tolist() if self.args.mode[0] != "1" else None
        X = self._encode_features(texts, img_paths, mode="train")

        perf_cols = [f"{m}_is_correct" for m in self.processor.model_list]
        cost_cols = [f"{m}_cost" for m in self.processor.model_list]
        Y_perf = np.nan_to_num(train_df[perf_cols].to_numpy(dtype=float), nan=0.0)
        Y_cost = np.nan_to_num(train_df[cost_cols].to_numpy(dtype=float), nan=0.0)

        rank = int(getattr(self.args, "mf_rank", 128))
        rank = max(1, min(rank, X.shape[0], X.shape[1]))
        self.svd = TruncatedSVD(n_components=rank, random_state=int(getattr(self.args, "random_state", 42)))
        Z = self.svd.fit_transform(X)

        alpha = float(getattr(self.args, "alpha", 1.0))
        self.reg_perf = MultiOutputRegressor(Ridge(alpha=alpha))
        self.reg_cost = MultiOutputRegressor(Ridge(alpha=alpha))
        self.reg_perf.fit(Z, Y_perf)
        self.reg_cost.fit(Z, Y_cost)

        test_df = self.processor.test_data
        test_texts = test_df["question"].astype(str).tolist()
        test_img = test_df["img_path"].tolist() if self.args.mode[1] != "1" else None
        X_test = self._encode_features(test_texts, test_img, mode="test")
        self.test_Z = self.svd.transform(X_test)

    def evaluate_under_cost(self):
        if self.processor is None:
            self.fit()

        perf_pred = self.reg_perf.predict(self.test_Z)
        cost_pred = self.reg_cost.predict(self.test_Z)
        n, _ = perf_pred.shape
        all_costs = np.sort(np.unique(cost_pred))
        test_df = self.processor.test_data
        row_idx = np.arange(n)
        points = []
        for C in all_costs:
            mask = cost_pred <= C
            masked_perf = np.where(mask, perf_pred, -np.inf)
            best_idx = np.argmax(masked_perf, axis=1)
            perf_cols = [self.processor.model_list[mid] + "_is_correct" for mid in best_idx]
            cost_cols = [self.processor.model_list[mid] + "_cost" for mid in best_idx]
            col_perf = [test_df.columns.get_loc(c) for c in perf_cols]
            col_cost = [test_df.columns.get_loc(c) for c in cost_cols]
            selected_perf = test_df.to_numpy()[row_idx, col_perf]
            selected_cost = test_df.to_numpy()[row_idx, col_cost]
            points.append({"cost": float(np.mean(selected_cost)), "performance": float(np.mean(selected_perf))})

        save_points(self.out_dir, self._name, self.args.dataset, self.args.mode, points)
        return points

