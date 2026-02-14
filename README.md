# MMR-Bench: A Comprehensive Benchmark for Multimodal LLM Routing

This folder contains the **open-source code release** for **MMR-Bench** (offline, cost-aware multimodal LLM routing).

---

## Overview

MMR-Bench is a comprehensive benchmark designed to evaluate **multimodal LLM routing** under diverse settings.  
It supports systematic comparison across scenarios and provides analysis beyond single-dataset evaluation (e.g., cost–accuracy trade-offs, cross-dataset generalization, and modality transfer).

<p align="center">
  <img src="assets/Bench3.jpg" alt="MMR-Bench overview" width="850">
</p>

---

## Comparison with Existing LLM Routing Benchmarks

<p align="center">
  <img src="assets/1.png" alt="Comparison: existing routing benchmarks vs MMR-Bench" width="700">
</p>

---

## Results

### Main Comparisons

<p align="center">
  <img src="assets/Result.png" alt="Main results on MMR-Bench" width="700">
</p>

### Cost–Accuracy Pareto Frontiers on MMR-Bench

<p align="center">
  <img src="assets/5.2.jpg" alt="Cost–accuracy Pareto frontiers on MMR-Bench" width="700">
</p>

### Within-Scenario Cross-Dataset Generalization

<p align="center">
  <img src="assets/Cross.png" alt="Within-scenario cross-dataset evaluation" width="520">
</p>

### Cross-Modality Transfer to Text-Only Benchmarks

<p align="center">
  <img src="assets/Modality.png" alt="Cross-modality transfer to text-only benchmarks" width="520">
</p>

---

## Installation

From this directory:

```bash
pip install -e .
```

Optional environment variables:

- `HF_HOME`: Hugging Face cache directory (defaults to `~/.cache/huggingface`).

If you want CLIP/OpenCLIP embeddings (text+image) for the baseline routers:

```bash
pip install -e '.[embedding]'
```

## Quickstart (toy data)

Generate a tiny synthetic offline benchmark (CSV + images), then run a router:

```bash
python scripts/make_toy_data.py
mmrbench --data-root data/toy --dataset toy --mode 22 --router kmeansnew
```

You can also run via module entrypoint:

```bash
python -m mmrbench --data-root data/toy --dataset toy --mode 22 --router kmeansnew
```

This writes a cost–accuracy curve to `outputs/` and prints a JSON summary including `nAUC`, `Ps`, and `QNC`.

## Real data (Hugging Face: `gh0stHunter/MMR-Bench`)

The full benchmark is distributed as:

- image folders (e.g. `MathVerse/`, `SEEDBenchv2Plus/`, …)
- a merged outcomes table `MMR_Bench.csv`

Place them under `data/` (see `data/README.md`). Example run:

```bash
python -m mmrbench --data-root data --dataset ocrbench+seedbench+mmstar --mode 22 --router linearmf
```

Optional: download helper script (requires HF access):

```bash
export HF_HOME=~/.cache/huggingface  # optional
pip install -e '.[hf]'
python scripts/prepare_hf_mmr_bench.py --dest data
```

## Data format

MMR-Bench is evaluated **offline**: for each instance and each candidate model, you provide:

- `question` (string)
- `img_path` (string; optional but recommended)
- for each model name `M`:
  - `M_correct` (0/1)
  - `M_cost` (float; any consistent cost unit)

You can route across multiple datasets by concatenating names with `+` (e.g. `ocrbench+mathvista`). By default the loader expects:

- CSV at `<data-root>/<dataset>.csv`
- optional images under `<data-root>/<dataset>/`

If `<data-root>/MMR_Bench.csv` exists and `--dataset` is a `+`-separated subset of:
`{ocrbench, seedbench, mmstar, realworldqa, mathvista, mathvision, mathverse}`,
the loader will use the merged CSV and infer image paths from `dataset_idx`.

## Reproducing paper numbers

This release focuses on the **routing algorithms + offline evaluation**. If you have the full MMR-Bench outcome tables from the paper (CSV/Parquet), point `--data-root` to them and run the corresponding routers/modes.

## Citation

If you find MMR-Bench useful, please consider citing our paper
