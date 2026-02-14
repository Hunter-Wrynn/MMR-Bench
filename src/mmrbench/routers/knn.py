from __future__ import annotations

import numpy as np
from pathlib import Path
from sklearn.neighbors import NearestNeighbors

from mmrbench.data import MMData
from mmrbench.embedding import Embedding
from mmrbench.routers._shared import save_points


class KNNRouter:
    def __init__(self, args):
        self.args = args
        self.out_dir = Path("./outputs")
        self.out_dir.mkdir(exist_ok=True, parents=True)
        self._name = "knn"
        self.embeder = Embedding(args)
        self.processor = None
        self.model_ids = None
        self.k = int(getattr(args, "knn_k", 50))
        self.metric = str(getattr(args, "knn_metric", "cosine"))
        self.X_train = None
        self.Y_perf = None
        self.Y_cost = None
        self.nn_perf = None
        self.nn_cost = None
        self.test_feats = None

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
        self.X_train = self._encode_features(texts, img_paths, mode="train")

        perf_cols = [f"{m}_is_correct" for m in self.processor.model_list]
        cost_cols = [f"{m}_cost" for m in self.processor.model_list]
        self.Y_perf = train_df[perf_cols].to_numpy(dtype=float)
        self.Y_cost = train_df[cost_cols].to_numpy(dtype=float)

        self.nn_perf = NearestNeighbors(n_neighbors=self.k, metric=self.metric)
        self.nn_cost = NearestNeighbors(n_neighbors=self.k, metric=self.metric)
        self.nn_perf.fit(self.X_train)
        self.nn_cost.fit(self.X_train)

        self._prepare_test()

    def _prepare_test(self):
        test_df = self.processor.test_data
        texts = test_df["question"].astype(str).tolist()
        img_paths = test_df["img_path"].tolist() if self.args.mode[1] != "1" else None
        self.test_feats = self._encode_features(texts, img_paths, mode="test")

    def _predict_knn_mean(self, X: np.ndarray, Y: np.ndarray, nn: NearestNeighbors) -> np.ndarray:
        _, idxs = nn.kneighbors(X, n_neighbors=self.k, return_distance=True)
        preds = np.full((X.shape[0], Y.shape[1]), np.nan, dtype=float)
        for i in range(X.shape[0]):
            neigh = Y[idxs[i]]
            preds[i] = np.nanmean(neigh, axis=0)
        return preds

    def evaluate_under_cost(self):
        if self.processor is None:
            self.fit()

        perf_pred = self._predict_knn_mean(self.test_feats, self.Y_perf, self.nn_perf)
        cost_pred = self._predict_knn_mean(self.test_feats, self.Y_cost, self.nn_cost)

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

