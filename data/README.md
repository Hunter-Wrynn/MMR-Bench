# Data layout

This repo evaluates routing **offline** using precomputed per-instance outcomes.

## Recommended layout (HF: `gh0stHunter/MMR-Bench`)

Place the merged CSV and image folders under `data/`:

```
data/
  MMR_Bench.csv
  MathVerse/
  MathVision/
  MathVista/
  OCRBench/
  RealWorldQA/
  MMStar/
  SEEDBenchv2Plus/
```

`MMR_Bench.csv` uses `dataset_idx` to reference images. Example:

- `SEEDBench2_Plus_0` → `data/SEEDBenchv2Plus/0.jpg`

## Download helper

If you have access to the Hugging Face dataset repo, you can run:

```bash
export HF_HOME=~/.cache/huggingface  # optional
python scripts/prepare_hf_mmr_bench.py --dest data
```
