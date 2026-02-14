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


class _MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden=(256, 128)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden[0]),
            nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
            nn.Linear(hidden[1], output_dim),
        )

    def forward(self, x):
        return self.net(x)


class MLPRouter:
    def __init__(self, args):
        self.args = args
        self.out_dir = Path("./outputs")
        self.out_dir.mkdir(exist_ok=True, parents=True)
        self._name = "mlp"
        self.embeder = Embedding(args)
        self.processor = None
        self.model_ids = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.perf_net = None
        self.cost_net = None
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
        Y_perf = np.nan_to_num(train_df[perf_cols].to_numpy(dtype=float), nan=0.0).astype(np.float32)
        Y_cost = np.nan_to_num(train_df[cost_cols].to_numpy(dtype=float), nan=0.0).astype(np.float32)

        input_dim = X.shape[1]
        out_dim = len(self.model_ids)
        hidden = tuple(getattr(self.args, "hidden_sizes", (256, 128)))
        self.perf_net = _MLP(input_dim, out_dim, hidden=hidden).to(self.device)
        self.cost_net = _MLP(input_dim, out_dim, hidden=hidden).to(self.device)

        epochs = int(getattr(self.args, "epochs", 50))
        batch = int(getattr(self.args, "batch_size", 32))
        lr = float(getattr(self.args, "lr", 1e-3))
        crit = nn.MSELoss()

        def train(net, Y):
            opt = optim.Adam(net.parameters(), lr=lr)
            ds = _ArrayDataset(X, Y)
            dl = DataLoader(ds, batch_size=batch, shuffle=True)
            net.train()
            for _ in range(epochs):
                for xb, yb in dl:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    opt.zero_grad()
                    pred = net(xb)
                    loss = crit(pred, yb)
                    loss.backward()
                    opt.step()

        train(self.perf_net, Y_perf)
        train(self.cost_net, Y_cost)

        test_df = self.processor.test_data
        test_texts = test_df["question"].astype(str).tolist()
        test_img = test_df["img_path"].tolist() if self.args.mode[1] != "1" else None
        self.test_X = self._encode_features(test_texts, test_img, mode="test")

    def evaluate_under_cost(self):
        if self.processor is None:
            self.fit()

        with torch.no_grad():
            X = torch.tensor(self.test_X, dtype=torch.float32).to(self.device)
            perf_pred = self.perf_net(X).cpu().numpy()
            cost_pred = self.cost_net(X).cpu().numpy()

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

