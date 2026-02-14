from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from mmrbench.data import MMData
from mmrbench.embedding import Embedding
from mmrbench.routers._shared import save_points


class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)
        ff = self.ff(x)
        return self.norm2(x + ff)


class CrossModalEncoder(nn.Module):
    def __init__(self, d_model: int, num_layers: int = 2, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [CrossAttentionLayer(d_model=d_model, num_heads=num_heads, dropout=dropout) for _ in range(max(1, num_layers))]
        )

    def forward(self, U):
        x = U
        for layer in self.layers:
            x = layer(x)
        return x


class CrossModalRouter(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.out_dir = Path("./outputs")
        self.out_dir.mkdir(exist_ok=True, parents=True)
        self._name = "cmr"
        self.embeder = Embedding(args)
        self.processor = None
        self.model_ids = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.d_model = int(getattr(args, "proj_dim", 512))
        self.num_layers = int(getattr(args, "num_layers", 2))
        self.num_heads = int(getattr(args, "num_heads", 8))
        self.hidden = int(getattr(args, "hidden_dim", 512))

        self.proj_text = None
        self.proj_img = None
        self.encoder = None
        self.head_perf = None
        self.head_cost = None

        self._raw = None
        self._test_raw = None

    def _encode_raw(self, texts, img_paths, mode="train"):
        mode_flag = self.args.mode[0] if mode == "train" else self.args.mode[1]
        X_text = self.embeder.run_text(texts).astype(np.float32)
        if mode_flag == "1":
            X_img = np.zeros_like(X_text)
        else:
            X_img = np.zeros_like(X_text) if img_paths is None else self.embeder.run_img(img_paths).astype(np.float32)
        return X_text, X_img

    def fit(self):
        self.processor = MMData(self.args)
        train_df = self.processor.train_data
        test_df = self.processor.test_data
        self.model_ids = list(self.processor.model_ids)
        K = len(self.model_ids)

        texts = train_df["question"].astype(str).tolist()
        img_paths = train_df["img_path"].tolist() if self.args.mode[0] != "1" else None
        X_text, X_img = self._encode_raw(texts, img_paths, mode="train")
        self._raw = (X_text, X_img)

        perf_cols = [f"{m}_is_correct" for m in self.processor.model_list]
        cost_cols = [f"{m}_cost" for m in self.processor.model_list]
        Y_perf = train_df[perf_cols].to_numpy(dtype=float).astype(np.float32)
        Y_cost = train_df[cost_cols].to_numpy(dtype=float).astype(np.float32)

        text_dim = X_text.shape[1]
        img_dim = X_img.shape[1]
        self.proj_text = nn.Linear(text_dim, self.d_model).to(self.device)
        self.proj_img = nn.Linear(img_dim, self.d_model).to(self.device)
        self.encoder = CrossModalEncoder(self.d_model, num_layers=self.num_layers, num_heads=self.num_heads).to(self.device)

        self.head_perf = nn.Sequential(
            nn.Linear(self.d_model, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, K),
        ).to(self.device)
        self.head_cost = nn.Sequential(
            nn.Linear(self.d_model, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, K),
        ).to(self.device)

        epochs = int(getattr(self.args, "epochs", 20))
        batch = int(getattr(self.args, "batch_size", 64))
        lr = float(getattr(self.args, "lr", 1e-4))

        opt = optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)

        X_text_t = torch.tensor(X_text, dtype=torch.float32)
        X_img_t = torch.tensor(X_img, dtype=torch.float32)
        Y_perf_t = torch.tensor(np.nan_to_num(Y_perf, nan=np.nan), dtype=torch.float32)
        Y_cost_t = torch.tensor(np.nan_to_num(Y_cost, nan=np.nan), dtype=torch.float32)

        self.train()
        n = X_text_t.shape[0]
        for _ in range(epochs):
            perm = torch.randperm(n)
            for s in range(0, n, batch):
                idx = perm[s : s + batch]
                t = X_text_t[idx].to(self.device)
                i = X_img_t[idx].to(self.device)
                yp = Y_perf_t[idx].to(self.device)
                yc = Y_cost_t[idx].to(self.device)

                opt.zero_grad()
                z = self._forward_latent(t, i)
                pred_p = self.head_perf(z)
                pred_c = self.head_cost(z)

                mask_p = ~torch.isnan(yp)
                mask_c = ~torch.isnan(yc)
                loss = 0.0
                if mask_p.sum() > 0:
                    ypf = torch.where(mask_p, yp, torch.zeros_like(yp))
                    diff = (pred_p - ypf) * mask_p.float()
                    loss = loss + (diff**2).sum() / mask_p.float().sum()
                if mask_c.sum() > 0:
                    ycf = torch.where(mask_c, yc, torch.zeros_like(yc))
                    diff = (pred_c - ycf) * mask_c.float()
                    loss = loss + (diff**2).sum() / mask_c.float().sum()
                loss.backward()
                opt.step()

        # cache test raw
        test_texts = test_df["question"].astype(str).tolist()
        test_img_paths = test_df["img_path"].tolist() if self.args.mode[1] != "1" else None
        self._test_raw = self._encode_raw(test_texts, test_img_paths, mode="test")

    def _forward_latent(self, raw_text, raw_img):
        t = self.proj_text(raw_text)
        i = self.proj_img(raw_img)
        U = torch.stack([t, i], dim=1)
        V = self.encoder(U)
        return V.mean(dim=1)

    def evaluate_under_cost(self):
        if self.processor is None:
            self.fit()

        X_text, X_img = self._test_raw
        with torch.no_grad():
            self.eval()
            t = torch.tensor(X_text, dtype=torch.float32).to(self.device)
            i = torch.tensor(X_img, dtype=torch.float32).to(self.device)
            z = self._forward_latent(t, i)
            perf_pred = self.head_perf(z).cpu().numpy()
            cost_pred = self.head_cost(z).cpu().numpy()

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

