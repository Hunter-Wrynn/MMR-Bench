from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunConfig:
    data_root: Path
    dataset: str
    split: str
    train_percent: float
    out_percent: float
    mode: str
    router: str
    random_state: int
    n_clusters: int
    knn_k: int
    knn_metric: str
    mf_rank: int
    alpha: float
    epochs: int
    batch_size: int
    lr: float
    hidden_sizes: tuple[int, int]
    latent_dim: int


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MMR-Bench: multimodal LLM routing (offline)")

    p.add_argument(
        "--data-root",
        type=Path,
        default=Path("./data"),
        help="Dataset root dir. Each dataset is expected at <data-root>/<name>.csv plus optional images under <data-root>/<name>/",
    )
    p.add_argument(
        "--dataset",
        type=str,
        default="toy",
        help="Dataset name(s). Use '+' to concatenate, e.g. 'ocrbench+mathvista'.",
    )
    p.add_argument("--split", type=str, default="in", choices=["in", "out"])
    p.add_argument("--train-percent", type=float, default=0.2)
    p.add_argument("--out-percent", type=float, default=0.2)

    p.add_argument(
        "--mode",
        type=str,
        default="22",
        choices=["11", "12", "13", "21", "22", "23", "31", "32", "33"],
        help="Train/test modality flags: 1=text, 2=multimodal, 3=image (e.g. 22=mm->mm, 12=text->mm).",
    )

    p.add_argument(
        "--router",
        type=str,
        default="linearmf",
        choices=["random", "oracle", "knn", "kmeans", "kmeansnew", "linear", "linearmf", "mlp", "mlpmf", "cmr"],
    )

    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--n-clusters", type=int, default=20)
    p.add_argument("--knn-k", type=int, default=50)
    p.add_argument("--knn-metric", type=str, default="cosine", choices=["cosine", "euclidean"])

    p.add_argument("--mf-rank", type=int, default=128)
    p.add_argument("--alpha", type=float, default=1.0)

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-sizes", type=int, nargs=2, default=(256, 128))
    p.add_argument("--latent-dim", type=int, default=64)

    # Embedding backends (optional; routers can run with precomputed embeddings only if you customize)
    p.add_argument("--text-embedding-backend", type=str, default="clip", choices=["clip", "tfidf", "openllm"])
    p.add_argument("--img-embedding-backend", type=str, default="clip", choices=["clip", "vitb16", "resnet32"])
    p.add_argument("--clip-model", type=str, default="ViT-B-32", help="OpenCLIP model name (e.g., ViT-B-32, ViT-L-14).")
    p.add_argument("--clip-pretrained", type=str, default="openai", help="OpenCLIP pretrained tag (e.g., openai, laion2b_s34b_b79k).")
    p.add_argument("--clip-batch-size", type=int, default=64)
    p.add_argument("--openllm-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")

    return p


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


def to_run_config(args: argparse.Namespace) -> RunConfig:
    return RunConfig(
        data_root=args.data_root,
        dataset=args.dataset,
        split=args.split,
        train_percent=args.train_percent,
        out_percent=args.out_percent,
        mode=args.mode,
        router=str(args.router).lower(),
        random_state=args.random_state,
        n_clusters=args.n_clusters,
        knn_k=args.knn_k,
        knn_metric=args.knn_metric,
        mf_rank=args.mf_rank,
        alpha=args.alpha,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_sizes=tuple(args.hidden_sizes),
        latent_dim=args.latent_dim,
    )

