from __future__ import annotations

from mmrbench.routers import (
    KMeansRouter,
    KMeansRouterNew,
    KNNRouter,
    LinearRouter,
    LinearMFRouter,
    MLPRouter,
    MLPMFRouter,
    OracleRouter,
    RandomRouter,
    CrossModalRouter,
)
from mmrbench.eval import RouterEvaluator


def build_router(args):
    r = str(args.router).lower()
    if r == "kmeans":
        return KMeansRouter(args)
    if r == "kmeansnew":
        return KMeansRouterNew(args)
    if r == "knn":
        return KNNRouter(args)
    if r == "linear":
        return LinearRouter(args)
    if r == "linearmf":
        return LinearMFRouter(args)
    if r == "mlp":
        return MLPRouter(args)
    if r == "mlpmf":
        return MLPMFRouter(args)
    if r == "random":
        return RandomRouter(args)
    if r == "oracle":
        return OracleRouter(args)
    if r == "cmr":
        return CrossModalRouter(args)
    raise ValueError(f"Unknown router: {args.router}")


def run_once(args) -> dict:
    router = build_router(args)
    router.fit()
    evaluator = RouterEvaluator(router)
    return evaluator.eval_all()

