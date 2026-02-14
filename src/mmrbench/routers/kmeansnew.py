from __future__ import annotations

import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans

from mmrbench.data import MMData
from mmrbench.embedding import Embedding
from mmrbench.routers._shared import save_points


class KMeansRouter:
    """
    KMeans router with adaptive fusion (prototype+norm confidence; sum/product/diff interactions).
    """

    def __init__(self, args):
        self.args = args
        self.out_dir = Path("./outputs")
        self.out_dir.mkdir(exist_ok=True, parents=True)
        self._name = "kmeansnew"
        self.K = int(getattr(args, "n_clusters", 20))
        self.embeder = Embedding(args)
        self.processor = None
        self.model_ids = None
        self.kmeans = None
        self.performance_repr = None
        self.cost_repr = None
        self.test_embs = None

    def _encode_features(self, texts, img_paths=None, mode="train"):
        eps = 1e-8
        mode_flag = self.args.mode[0] if mode == "train" else self.args.mode[1]
        if mode_flag == "1":
            return self.embeder.run_text(texts).astype(np.float32)

        X_text = self.embeder.run_text(texts).astype(np.float32)
        X_img = np.zeros_like(X_text) if img_paths is None else self.embeder.run_img(img_paths).astype(np.float32)
        if X_text.shape[1] != X_img.shape[1]:
            d = min(X_text.shape[1], X_img.shape[1])
            X_text = X_text[:, :d]
            X_img = X_img[:, :d]

        mu_text = np.mean(X_text, axis=0, keepdims=True)
        mu_img = np.mean(X_img, axis=0, keepdims=True)

        def cosine_sim_rows(A, B):
            num = np.sum(A * B, axis=1)
            denom = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + eps
            return num / denom

        s_text = cosine_sim_rows(X_text, mu_text)
        s_img = cosine_sim_rows(X_img, mu_img)
        c_text_proto = (s_text + 1.0) / 2.0
        c_img_proto = (s_img + 1.0) / 2.0

        norms_text = np.linalg.norm(X_text, axis=1)
        norms_img = np.linalg.norm(X_img, axis=1)
        mean_text, std_text = float(np.mean(norms_text)), float(np.std(norms_text) + eps)
        mean_img, std_img = float(np.mean(norms_img)), float(np.std(norms_img) + eps)

        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-x))

        c_text_norm = sigmoid((norms_text - mean_text) / std_text)
        c_img_norm = sigmoid((norms_img - mean_img) / std_img)
        c_text = np.clip(0.5 * c_text_proto + 0.5 * c_text_norm, 0.0, 1.0)
        c_img = np.clip(0.5 * c_img_proto + 0.5 * c_img_norm, 0.0, 1.0)

        eta = 5.0
        exp_text = np.exp(eta * c_text)
        exp_img = np.exp(eta * c_img)
        denom = exp_text + exp_img + eps
        w_text = (exp_text / denom)[:, None]
        w_img = (exp_img / denom)[:, None]

        e_w = w_text * X_text + w_img * X_img
        e_odot = X_text * X_img
        e_delta = np.abs(X_text - X_img)
        f_raw = e_w + 0.5 * e_odot + 0.5 * e_delta
        f_raw = f_raw / (np.linalg.norm(f_raw, axis=1, keepdims=True) + eps)
        return f_raw.astype(np.float32)

    def fit(self):
        self.processor = MMData(self.args)
        train_df = self.processor.train_data
        test_df = self.processor.test_data
        self.model_ids = list(self.processor.model_ids)

        texts = train_df["question"].astype(str).tolist()
        img_paths = train_df["img_path"].tolist() if self.args.mode[0] != "1" else None
        X = self._encode_features(texts, img_paths, mode="train")

        perf_cols = [f"{m}_is_correct" for m in self.processor.model_list]
        cost_cols = [f"{m}_cost" for m in self.processor.model_list]
        Y_perf = train_df[perf_cols].to_numpy(dtype=float)
        Y_cost = train_df[cost_cols].to_numpy(dtype=float)

        self.kmeans = KMeans(n_clusters=self.K, random_state=int(getattr(self.args, "random_state", 42)), n_init="auto")
        cluster_ids = self.kmeans.fit_predict(X)

        n_models = len(self.model_ids)
        perf_sum = np.zeros((self.K, n_models), dtype=float)
        cost_sum = np.zeros((self.K, n_models), dtype=float)
        perf_cnt = np.zeros((self.K, n_models), dtype=int)
        cost_cnt = np.zeros((self.K, n_models), dtype=int)

        for i, c in enumerate(cluster_ids):
            for j in range(n_models):
                v = Y_perf[i, j]
                if not np.isnan(v):
                    perf_sum[c, j] += v
                    perf_cnt[c, j] += 1
                vc = Y_cost[i, j]
                if not np.isnan(vc):
                    cost_sum[c, j] += vc
                    cost_cnt[c, j] += 1

        with np.errstate(divide="ignore", invalid="ignore"):
            self.performance_repr = np.where(perf_cnt > 0, perf_sum / perf_cnt, np.nan)
            self.cost_repr = np.where(cost_cnt > 0, cost_sum / cost_cnt, np.nan)

        test_texts = test_df["question"].astype(str).tolist()
        test_img_paths = test_df["img_path"].tolist() if self.args.mode[1] != "1" else None
        self.test_embs = self._encode_features(test_texts, test_img_paths, mode="test")

    def evaluate_under_cost(self):
        if self.processor is None:
            self.fit()

        test_df = self.processor.test_data
        cluster_ids = self.kmeans.predict(self.test_embs)
        perf_pred = self.performance_repr[cluster_ids]
        cost_pred = self.cost_repr[cluster_ids]

        n, _ = perf_pred.shape
        all_costs = np.sort(np.unique(cost_pred))
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

