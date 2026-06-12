#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

import pandas as pd

from vlmeval.vlm.qwen2_vl.model import Qwen2VLChat


def smi(tag: str) -> None:
    print(f"[{tag}] nvidia-smi", flush=True)
    subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test Qwen2.5-VL-72B with VLMEvalKit vLLM adapter.")
    parser.add_argument("--model-path", default="/root/storage/mahaoxuan.mhx/model/Qwen2.5-VL-72B-Instruct")
    parser.add_argument("--csv", default="data/MMR-Bench.csv")
    parser.add_argument("--dataset-idx", default="MathVision_2677")
    parser.add_argument("--dataset", default="MathVision")
    parser.add_argument("--image", default="/tmp/mmr_smoke_images/MathVision/2677.jpg")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--min-pixels", type=int, default=1003520)
    parser.add_argument("--max-pixels", type=int, default=12845056)
    parser.add_argument("--gpu-util", type=float, default=0.85)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"VLLM_SMOKE_START model={args.model_path} cuda_visible={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    print(
        "config=use_vllm=True tensor_parallel_size=auto "
        f"gpu_memory_utilization={args.gpu_util} max_new_tokens={args.max_new_tokens} "
        f"min_pixels={args.min_pixels} max_pixels={args.max_pixels}",
        flush=True,
    )
    smi("before_load")

    start = time.time()
    try:
        model = Qwen2VLChat(
            model_path=args.model_path,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            top_k=1,
            repetition_penalty=1.0,
            use_custom_prompt=False,
            attn_implementation="sdpa",
            use_vllm=True,
            gpu_utils=args.gpu_util,
        )
    except Exception as exc:
        print(f"VLLM_LOAD_FAIL type={type(exc).__name__} message={exc}", flush=True)
        smi("after_load_fail")
        raise

    print(f"VLLM_LOAD_OK seconds={time.time() - start:.1f}", flush=True)
    smi("after_load")

    df = pd.read_csv(args.csv, usecols=["dataset_idx", "question", "answer"])
    row = df[df["dataset_idx"] == args.dataset_idx].iloc[0]
    question = str(row["question"])
    answer = str(row["answer"])
    image_path = Path(args.image)
    print(
        f"CASE_START {args.dataset_idx} dataset={args.dataset} "
        f"image_exists={image_path.exists()} question_chars={len(question)} answer={answer}",
        flush=True,
    )

    case_start = time.time()
    try:
        response = model.generate(
            message=[
                {"type": "image", "value": str(image_path)},
                {"type": "text", "value": question},
            ],
            dataset=args.dataset,
        )
    except Exception as exc:
        print(f"CASE_FAIL {args.dataset_idx} type={type(exc).__name__} message={exc}", flush=True)
        smi(f"after_{args.dataset_idx}_fail")
        raise

    try:
        response_tokens = len(model.processor.tokenizer.encode(response, add_special_tokens=False))
    except Exception:
        response_tokens = -1

    print(f"CASE_OK {args.dataset_idx} seconds={time.time() - case_start:.1f} response_tokens={response_tokens}", flush=True)
    print("RESPONSE_HEAD " + response[:1000].replace("\n", " "), flush=True)
    smi(f"after_{args.dataset_idx}")
    print("VLLM_SMOKE_OK all_cases_passed", flush=True)


if __name__ == "__main__":
    main()
