from __future__ import annotations

import json
from pathlib import Path

from mmrbench.config import parse_args
from mmrbench.run import run_once


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = run_once(args)
    out = json.dumps(result, ensure_ascii=False, indent=2)
    print(out)

