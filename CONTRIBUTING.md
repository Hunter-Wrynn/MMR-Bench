# Contributing

Thanks for your interest in contributing to MMR-Bench.

## What to contribute

- Bug fixes and performance improvements for routers and evaluation.
- New routing policies that operate on offline outcome tables.
- Better data adapters to common benchmark formats.

## Development setup

```bash
pip install -e '.[embedding]'
python scripts/make_toy_data.py
mmrbench --data-root data/toy --dataset toy --mode 22 --router kmeansnew
```

## Pull requests

- Keep changes focused and well-scoped.
- Add a short note in the PR description on how you validated the change (commands + expected output).

