#!/usr/bin/env python3
import argparse

from vlmeval.vlm.ovis.ovis import Ovis2_5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--image", default="third_party/VLMEvalKit/assets/apple.jpg")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.55)
    args = parser.parse_args()

    model = Ovis2_5(
        model_path=args.model_path,
        enable_thinking=False,
        max_new_tokens=args.max_new_tokens,
        thinking_budget=0,
        min_pixels=448 * 448,
        max_pixels=1792 * 1792,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    message = [
        {"type": "image", "value": args.image},
        {"type": "text", "value": "Answer briefly: what is in the image?"},
    ]
    print(model.generate(message, dataset="MMStar"))


if __name__ == "__main__":
    main()
