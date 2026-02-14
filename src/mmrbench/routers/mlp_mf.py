from __future__ import annotations

import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from mmrbench.data import MMData
from mmrbench.embedding import Embedding
from mmrbench.routers._shared import save_points


class _ArrayDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class _FeatureMLP(nn.Module):
    def __init__(self, input_dim: int, hidden=(256, 128), latent_dim=64):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MLPMFRouter:
    def __init__(self, args):
        self.args = args
        self.out_dir = Path("./outputs")
        self.out_dir.mkdir(exist_ok=True, parents=True)
        self._name = "mlpmf"
        self.embeder = Embedding(args)
        self.processor = None
        self.model_ids = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.perf_net = None
        self.perf_W = None
        self.perf_b = None

        self.cost_net = None
        self.cost_W = None
        self.cost_b = None

        self.test_X = None

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
        Y_perf = train_df[perf_cols].to_numpy(dtype=float).astype(np.float32)
        Y_cost = train_df[cost_cols].to_numpy(dtype=float).astype(np.float32)

        n_models = len(self.model_ids)
        input_dim = X.shape[1]
        hidden = tuple(getattr(self.args, "hidden_sizes", (256, 128)))
        latent_dim = int(getattr(self.args, "latent_dim", 64))

        self.perf_net = _FeatureMLP(input_dim, hidden=hidden, latent_dim=latent_dim).to(self.device)
        self.cost_net = _FeatureMLP(input_dim, hidden=hidden, latent_dim=latent_dim).to(self.device)
        self.perf_W = nn.Parameter(torch.randn(n_models, latent_dim, device=self.device) * 0.01)
        self.perf_b = nn.Parameter(torch.zeros(n_models, device=self.device))
        self.cost_W = nn.Parameter(torch.randn(n_models, latent_dim, device=self.device) * 0.01)
        self.cost_b = nn.Parameter(torch.zeros(n_models, device=self.device))

        epochs = int(getattr(self.args, "epochs", 50))
        batch = int(getattr(self.args, "batch_size", 32))
        lr = float(getattr(self.args, "lr", 1e-3))

        def train(feature_net, W, b, Y):
            ds = _ArrayDataset(X, np.nan_to_num(Y, nan=np.nan).astype(np.float32))
            dl = DataLoader(ds, batch_size=batch, shuffle=True)
            opt = optim.Adam(list(feature_net.parameters()) + [W, b], lr=lr)
            feature_net.train()
            for _ in range(epochs):
                for xb, yb in dl:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    opt.zero_grad()
                    z = feature_net(xb)
                    pred = z @ W.t() + b.unsqueeze(0)
                    mask = ~torch.isnan(yb)
                    if mask.sum() == 0:
                        continue
                    y_fill = torch.where(mask, yb, torch.zeros_like(yb))
                    diff = (pred - y_fill) * mask.float()
                    loss = (diff**2).sum() / mask.float().sum()
                    loss.backward()
                    opt.step()

        train(self.perf_net, self.perf_W, self.perf_b, Y_perf)
        train(self.cost_net, self.cost_W, self.cost_b, Y_cost)

        test_df = self.processor.test_data
        test_texts = test_df["question"].astype(str).tolist()
        test_img = test_df["img_path"].tolist() if self.args.mode[1] != "1" else None
        self.test_X = self._encode_features(test_texts, test_img, mode="test")

    def evaluate_under_cost(self):
        if self.processor is None:
            self.fit()

        with torch.no_grad():
            X = torch.tensor(self.test_X, dtype=torch.float32).to(self.device)
            zp = self.perf_net(X)
            zc = self.cost_net(X)
            perf_pred = (zp @ self.perf_W.t() + self.perf_b.unsqueeze(0)).cpu().numpy()
            cost_pred = (zc @ self.cost_W.t() + self.cost_b.unsqueeze(0)).cpu().numpy()

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

