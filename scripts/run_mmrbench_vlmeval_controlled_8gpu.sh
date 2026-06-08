#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLMEVAL_DIR="${REPO_ROOT}/third_party/VLMEvalKit"
DEFAULT_CONFIG_FILE="${REPO_ROOT}/configs/internvl35_4b_mmrbench_controlled.env"
CONFIG_FILE="${MMR_CONFIG_FILE:-${DEFAULT_CONFIG_FILE}}"
DEFAULT_MMR_BENCHMARKS="MMStar,RealWorldQA,SEEDBench2_Plus,OCRBench,MathVista_MINI,MathVision,MathVerse_MINI_Vision_Only"

ARGS=("$@")
for ((i = 0; i < ${#ARGS[@]}; i++)); do
  if [[ "${ARGS[$i]}" == "--config-file" ]]; then
    if [[ $((i + 1)) -ge ${#ARGS[@]} ]]; then
      echo "--config-file requires a path" >&2
      exit 2
    fi
    CONFIG_FILE="${ARGS[$((i + 1))]}"
  fi
done

if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "Missing runtime config: ${CONFIG_FILE}" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "${CONFIG_FILE}"

CONDA_ENV="${VLMEVAL_CONDA_ENV:-${CONDA_ENV:-/root/storage/miniconda3/envs/vlmevalkit}}"

MODEL="${MMR_MODEL:-${MODEL:-}}"
MODEL_PATH="${MMR_MODEL_PATH:-${MODEL_PATH:-}}"
MODEL_CLASS="${MMR_MODEL_CLASS:-${MODEL_CLASS:-}}"
DEFAULT_MODEL_ARGS_JSON='{}'
MODEL_ARGS_JSON="${MODEL_ARGS_JSON:-${DEFAULT_MODEL_ARGS_JSON}}"
MODEL_ARGS_JSON="${MMR_MODEL_ARGS_JSON:-${MODEL_ARGS_JSON}}"
CSV_PATH="${MMR_CSV_PATH:-${CSV_PATH:-${REPO_ROOT}/data/MMR-Bench.csv}}"

BENCHMARK_REGISTRY="${MMR_BENCHMARK_REGISTRY:-${BENCHMARK_REGISTRY:-${REPO_ROOT}/configs/mmrbench_benchmark_registry.json}}"
BENCHMARKS="${MMR_BENCHMARKS:-${BENCHMARKS:-${DEFAULT_MMR_BENCHMARKS}}}"
MERGE_MODE="${MMR_MERGE_MODE:-${MERGE_MODE:-auto}}"

JUDGE_MODEL="${MMR_JUDGE_MODEL:-${JUDGE_MODEL:-Qwen3.5-122B-A10B}}"
JUDGE_IP="${MMR_JUDGE_IP:-${JUDGE_IP:-33.3.178.31}}"
JUDGE_PORT="${MMR_JUDGE_PORT:-${JUDGE_PORT:-8000}}"
JUDGE_BASE_URL="${MMR_JUDGE_BASE_URL:-${JUDGE_BASE_URL:-}}"
JUDGE_KEY="${MMR_JUDGE_KEY:-${JUDGE_KEY:-EMPTY}}"
JUDGE_NPROC="${MMR_JUDGE_NPROC:-${JUDGE_NPROC:-8}}"
JUDGE_RETRY="${MMR_JUDGE_RETRY:-${JUDGE_RETRY:-2}}"
JUDGE_TIMEOUT="${MMR_JUDGE_TIMEOUT:-${JUDGE_TIMEOUT:-900}}"
DEFAULT_JUDGE_ARGS='{"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}'
JUDGE_ARGS="${JUDGE_ARGS:-${DEFAULT_JUDGE_ARGS}}"
JUDGE_ARGS="${MMR_JUDGE_ARGS:-${JUDGE_ARGS}}"

GPUS="${MMR_GPUS:-${GPUS:-0,1,2,3,4,5,6,7}}"
NPROC="${MMR_NPROC:-${NPROC:-}}"
MASTER_PORT="${MMR_MASTER_PORT:-${MASTER_PORT:-29551}}"
USE_COT="${MMR_USE_COT:-${USE_COT:-1}}"
API_NPROC="${MMR_API_NPROC:-${API_NPROC:-1}}"
PREWARM_REMOTE_CODE="${MMR_PREWARM_REMOTE_CODE:-${PREWARM_REMOTE_CODE:-1}}"

TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
RUN_NAME="${MMR_RUN_NAME:-${RUN_NAME:-mmrbench_controlled_${MODEL:-model}_${TIMESTAMP}}}"
WORK_DIR="${MMR_WORK_DIR:-${WORK_DIR:-}}"
LOG_DIR="${MMR_LOG_DIR:-${LOG_DIR:-${REPO_ROOT}/logs}}"

DRY_RUN=0
MERGE_REQUESTED=""
DO_BACKUP=1

usage() {
  cat <<'EOF'
Usage:
  scripts/run_mmrbench_vlmeval_controlled_8gpu.sh [options]

Generic VLMEvalKit runner for MMR-Bench controlled main-table runs.
It generates a VLMEvalKit config from a model spec and an explicit benchmark list,
runs inference/evaluation with torchrun sharding, logs metadata, and optionally
merges the standard 7 MMR-Bench results back into data/MMR-Bench.csv.

Common examples:
  # Run the default InternVL3.5-4B config on the 7 MMR-Bench benchmarks
  scripts/run_mmrbench_vlmeval_controlled_8gpu.sh \
    --benchmarks MMStar,RealWorldQA,SEEDBench2_Plus,OCRBench,MathVista_MINI,MathVision,MathVerse_MINI_Vision_Only

  # Switch model without editing the script
  scripts/run_mmrbench_vlmeval_controlled_8gpu.sh \
    --model MyModel --model-path /path/to/model --model-class InternVLChat \
    --model-args-json '{"version":"V2.0","device_map":null,"max_new_tokens":4096,"do_sample":false}'

Options:
  --config-file PATH      Runtime config file. Default: configs/internvl35_4b_mmrbench_controlled.env
  --model NAME            VLMEvalKit model key / output column prefix.
  --model-path PATH       Local or HF model path.
  --model-class CLASS     VLMEvalKit class, e.g. InternVLChat, Qwen2VLChat, Qwen3VL.
  --model-args-json JSON  Extra model kwargs JSON merged into the generated config.
  --benchmarks LIST       Required benchmark list, comma-separated or space-separated.
  --benchmark-registry PATH
                          JSON registry mapping benchmark names to VLMEvalKit class/dataset.
  --judge-ip IP           Judge host IP. Default comes from config.
  --judge-port PORT       Judge port. Default: 8000.
  --judge-base-url URL    Full judge base URL, e.g. http://33.3.178.31:8000/v1.
  --judge-model NAME      Judge model name. Default: Qwen3.5-122B-A10B.
  --judge-key KEY         Judge API key. Default: EMPTY.
  --gpus LIST             CUDA_VISIBLE_DEVICES list. Default: 0,1,2,3,4,5,6,7.
  --nproc N               torchrun nproc-per-node. Default: number of GPUs in --gpus.
  --master-port PORT      torchrun master port.
  --work-dir DIR          VLMEvalKit output directory root.
  --csv PATH              MMR-Bench CSV to update.
  --run-name NAME         Name used for default output/log paths.
  --use-cot 0|1           Set USE_COT. Default comes from config.
  --no-prewarm-remote-code
                          Disable local trust_remote_code cache prewarm before torchrun.
  --merge                 Force merge. Only valid for the standard 7 MMR-Bench benchmarks.
  --no-merge              Do not merge results back into the MMR-Bench CSV.
  --no-backup             Do not create data/MMR-Bench.before_<model>.csv.
  --dry-run               Generate config/metadata and print command, then exit.
  -h, --help              Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config-file) CONFIG_FILE="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --model-class) MODEL_CLASS="$2"; shift 2 ;;
    --model-args-json) MODEL_ARGS_JSON="$2"; shift 2 ;;
    --benchmarks) BENCHMARKS="$2"; shift 2 ;;
    --benchmark-registry) BENCHMARK_REGISTRY="$2"; shift 2 ;;
    --judge-ip) JUDGE_IP="$2"; shift 2 ;;
    --judge-port) JUDGE_PORT="$2"; shift 2 ;;
    --judge-base-url) JUDGE_BASE_URL="$2"; shift 2 ;;
    --judge-model) JUDGE_MODEL="$2"; shift 2 ;;
    --judge-key) JUDGE_KEY="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --nproc) NPROC="$2"; shift 2 ;;
    --master-port) MASTER_PORT="$2"; shift 2 ;;
    --work-dir) WORK_DIR="$2"; shift 2 ;;
    --csv) CSV_PATH="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --use-cot) USE_COT="$2"; shift 2 ;;
    --no-prewarm-remote-code) PREWARM_REMOTE_CODE=0; shift ;;
    --merge) MERGE_REQUESTED=1; shift ;;
    --no-merge) MERGE_REQUESTED=0; shift ;;
    --no-backup) DO_BACKUP=0; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "${WORK_DIR}" ]]; then
  WORK_DIR="${VLMEVAL_DIR}/outputs/${RUN_NAME}"
fi
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"
META_FILE="${LOG_DIR}/${RUN_NAME}.meta.json"
GENERATED_CONFIG="${WORK_DIR}/${RUN_NAME}.vlmeval_config.json"

if [[ -z "${MODEL}" ]]; then
  echo "Missing model name. Set MODEL in config or pass --model." >&2
  exit 2
fi
if [[ -z "${MODEL_PATH}" ]]; then
  echo "Missing model path. Set MODEL_PATH in config or pass --model-path." >&2
  exit 2
fi
if [[ -z "${MODEL_CLASS}" ]]; then
  echo "Missing model class. Set MODEL_CLASS in config or pass --model-class." >&2
  exit 2
fi
if [[ -z "${BENCHMARKS//[[:space:],]/}" ]]; then
  echo "Missing benchmarks. Set BENCHMARKS in config or pass --benchmarks." >&2
  exit 2
fi
if [[ -z "${JUDGE_BASE_URL}" ]]; then
  JUDGE_BASE_URL="http://${JUDGE_IP}:${JUDGE_PORT}/v1"
fi
if [[ -z "${NPROC}" ]]; then
  IFS=',' read -r -a GPU_ARRAY <<< "${GPUS}"
  NPROC="${#GPU_ARRAY[@]}"
fi
if [[ "${NPROC}" -lt 1 ]]; then
  echo "Invalid --nproc: ${NPROC}" >&2
  exit 2
fi

if [[ ! -x "${CONDA_ENV}/bin/python" ]]; then
  echo "Missing python in conda env: ${CONDA_ENV}" >&2
  exit 1
fi
if [[ ! -f "${BENCHMARK_REGISTRY}" ]]; then
  echo "Missing benchmark registry: ${BENCHMARK_REGISTRY}" >&2
  exit 1
fi
if [[ "${MODEL_PATH}" == /* || "${MODEL_PATH}" == ./* || "${MODEL_PATH}" == ../* ]]; then
  if [[ ! -e "${MODEL_PATH}" ]]; then
    echo "Missing model path: ${MODEL_PATH}" >&2
    exit 1
  fi
fi

mkdir -p "${LOG_DIR}" "${WORK_DIR}"

BENCHMARKS_CANON="$("${CONDA_ENV}/bin/python" - "$BENCHMARKS" <<'PY'
import re
import sys
items = [x for x in re.split(r'[\s,]+', sys.argv[1].strip()) if x]
print(','.join(items))
PY
)"

generate_vlmeval_config() {
  "${CONDA_ENV}/bin/python" - \
    "$GENERATED_CONFIG" "$MODEL" "$MODEL_CLASS" "$MODEL_PATH" "$MODEL_ARGS_JSON" \
    "$BENCHMARK_REGISTRY" "$BENCHMARKS_CANON" <<'PY'
import json
import pathlib
import sys

out_path = pathlib.Path(sys.argv[1])
model_name, model_class, model_path = sys.argv[2], sys.argv[3], sys.argv[4]
model_args = json.loads(sys.argv[5])
registry = json.loads(pathlib.Path(sys.argv[6]).read_text())
benchmarks = [x for x in sys.argv[7].split(',') if x]

if not isinstance(model_args, dict):
    raise SystemExit("MODEL_ARGS_JSON must decode to a JSON object")

missing = [name for name in benchmarks if name not in registry]
if missing:
    known = ', '.join(sorted(registry))
    raise SystemExit(f"Unknown benchmark(s): {missing}. Known benchmarks: {known}")

model_cfg = dict(model_args)
model_cfg["class"] = model_class
model_cfg["model_path"] = model_path

config = {
    "model": {model_name: model_cfg},
    "data": {name: registry[name] for name in benchmarks},
}
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
print(out_path)
PY
}

generate_vlmeval_config >/dev/null

if [[ ! -f "${CSV_PATH}" && "${MERGE_REQUESTED:-${MERGE_MODE}}" != "0" && "${MERGE_MODE}" != "never" ]]; then
  echo "Missing CSV for merge: ${CSV_PATH}" >&2
  exit 1
fi

MERGE_ALLOWED="$("${CONDA_ENV}/bin/python" - "$BENCHMARKS_CANON" "$DEFAULT_MMR_BENCHMARKS" <<'PY'
import sys
selected = set(x for x in sys.argv[1].split(',') if x)
default = set(x for x in sys.argv[2].split(',') if x)
print("1" if selected == default else "0")
PY
)"

if [[ "${MERGE_REQUESTED}" == "1" ]]; then
  if [[ "${MERGE_ALLOWED}" != "1" ]]; then
    echo "--merge is only supported for the standard 7 MMR-Bench benchmarks. Use --no-merge for custom benchmark sets." >&2
    exit 2
  fi
  DO_MERGE=1
elif [[ "${MERGE_REQUESTED}" == "0" || "${MERGE_MODE}" == "never" ]]; then
  DO_MERGE=0
elif [[ "${MERGE_MODE}" == "always" ]]; then
  DO_MERGE=1
else
  DO_MERGE="${MERGE_ALLOWED}"
fi

JUDGE_HOST="${JUDGE_BASE_URL#http://}"
JUDGE_HOST="${JUDGE_HOST#https://}"
JUDGE_HOST="${JUDGE_HOST%%/*}"
JUDGE_HOST="${JUDGE_HOST%%:*}"
NO_PROXY_VALUE="${NO_PROXY:-${no_proxy:-}}"
for host in localhost 127.0.0.1 ::1 "${JUDGE_HOST}"; do
  if [[ ",${NO_PROXY_VALUE}," != *",${host},"* ]]; then
    NO_PROXY_VALUE="${NO_PROXY_VALUE:+${NO_PROXY_VALUE},}${host}"
  fi
done

write_metadata() {
  "${CONDA_ENV}/bin/python" - \
    "$META_FILE" "$RUN_NAME" "$CONFIG_FILE" "$MODEL" "$MODEL_PATH" "$MODEL_CLASS" \
    "$MODEL_ARGS_JSON" "$GENERATED_CONFIG" "$BENCHMARK_REGISTRY" "$BENCHMARKS_CANON" \
    "$CSV_PATH" "$WORK_DIR" "$LOG_FILE" "$USE_COT" "$JUDGE_MODEL" "$JUDGE_BASE_URL" \
    "$JUDGE_NPROC" "$JUDGE_RETRY" "$JUDGE_TIMEOUT" "$JUDGE_ARGS" "$GPUS" "$NPROC" \
    "$MASTER_PORT" "$DO_MERGE" <<'PY'
import json
import pathlib
import sys

(
    meta_path, run_name, runtime_config, model, model_path, model_class,
    model_args_json, generated_config, benchmark_registry, benchmarks,
    csv_path, work_dir, log_file, use_cot, judge_model, judge_base_url,
    judge_nproc, judge_retry, judge_timeout, judge_args, gpus, nproc,
    master_port, do_merge,
) = sys.argv[1:]

meta = {
    "run_name": run_name,
    "runtime_config": runtime_config,
    "model": model,
    "model_path": model_path,
    "model_class": model_class,
    "model_args": json.loads(model_args_json),
    "generated_vlmeval_config": generated_config,
    "benchmark_registry": benchmark_registry,
    "benchmarks": [x for x in benchmarks.split(',') if x],
    "csv_path": csv_path,
    "work_dir": work_dir,
    "log_file": log_file,
    "protocol": "MMR-Bench controlled main-table v2",
    "inference": {
        "mode": "non-thinking",
        "use_cot": use_cot,
        "pred_format": "xlsx",
    },
    "judge": {
        "model": judge_model,
        "base_url": judge_base_url,
        "nproc": int(judge_nproc),
        "retry": int(judge_retry),
        "timeout": int(judge_timeout),
        "args": json.loads(judge_args),
    },
    "parallel": {
        "gpus": gpus,
        "nproc": int(nproc),
        "master_port": int(master_port),
    },
    "merge_to_mmr_csv": bool(int(do_merge)),
}
path = pathlib.Path(meta_path)
path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
print(path)
PY
}

write_metadata >/dev/null

CMD=(
  "${CONDA_ENV}/bin/python" -m torch.distributed.run
  "--nproc-per-node=${NPROC}"
  "--master-port=${MASTER_PORT}"
  run.py
  --config "${GENERATED_CONFIG}"
  --judge "${JUDGE_MODEL}"
  --judge-base-url "${JUDGE_BASE_URL}"
  --judge-key "${JUDGE_KEY}"
  --judge-api-nproc "${JUDGE_NPROC}"
  --judge-retry "${JUDGE_RETRY}"
  --judge-timeout "${JUDGE_TIMEOUT}"
  --judge-args "${JUDGE_ARGS}"
  --api-nproc "${API_NPROC}"
  --work-dir "${WORK_DIR}"
)

echo "run_name=${RUN_NAME}"
echo "runtime_config=${CONFIG_FILE}"
echo "generated_vlmeval_config=${GENERATED_CONFIG}"
echo "model=${MODEL}"
echo "model_class=${MODEL_CLASS}"
echo "benchmarks=${BENCHMARKS_CANON}"
echo "judge_base_url=${JUDGE_BASE_URL}"
echo "gpus=${GPUS}"
echo "nproc=${NPROC}"
echo "merge_to_mmr_csv=${DO_MERGE}"
echo "work_dir=${WORK_DIR}"
echo "log_file=${LOG_FILE}"
echo "metadata=${META_FILE}"
printf 'command='
printf ' %q' CUDA_VISIBLE_DEVICES="${GPUS}" USE_COT="${USE_COT}" PRED_FORMAT=xlsx NO_PROXY="${NO_PROXY_VALUE}" "${CMD[@]}"
printf '\n'

if [[ "${DRY_RUN}" -eq 1 ]]; then
  exit 0
fi

prewarm_remote_code_cache() {
  if [[ "${PREWARM_REMOTE_CODE}" != "1" ]]; then
    return 0
  fi
  if [[ ! -d "${MODEL_PATH}" ]]; then
    return 0
  fi
  if ! compgen -G "${MODEL_PATH}/*.py" >/dev/null; then
    return 0
  fi

  "${CONDA_ENV}/bin/python" - "$MODEL_PATH" "$MODEL_ARGS_JSON" <<'PY'
import filecmp
import importlib
import json
import shutil
import sys
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module
try:
    from transformers.dynamic_module_utils import _sanitize_module_name
except Exception:
    def _sanitize_module_name(name):
        return name.replace("-", "_hyphen_").replace(".", "_dot_")
from transformers.utils import HF_MODULES_CACHE, TRANSFORMERS_DYNAMIC_MODULE_NAME

model_path = Path(sys.argv[1]).resolve()
model_args = json.loads(sys.argv[2])
tokenizer_kwargs = {
    "trust_remote_code": True,
    "use_fast": False,
}
if "fix_mistral_regex" in model_args:
    tokenizer_kwargs["fix_mistral_regex"] = model_args["fix_mistral_regex"]

print(f"prewarm_remote_code_cache=model_path:{model_path}")
AutoTokenizer.from_pretrained(str(model_path), **tokenizer_kwargs)
cfg = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)

auto_map = getattr(cfg, "auto_map", None) or {}
if not auto_map:
    config_json = model_path / "config.json"
    if config_json.exists():
        auto_map = json.loads(config_json.read_text()).get("auto_map", {}) or {}

class_refs = []
for value in auto_map.values():
    if isinstance(value, str) and "." in value:
        class_refs.append(value)
    elif isinstance(value, (list, tuple)):
        class_refs.extend(x for x in value if isinstance(x, str) and "." in x)

for ref in sorted(set(class_refs)):
    try:
        get_class_from_dynamic_module(ref, str(model_path), local_files_only=True)
    except FileNotFoundError:
        pass

cache_root = (
    Path(HF_MODULES_CACHE)
    / TRANSFORMERS_DYNAMIC_MODULE_NAME
    / _sanitize_module_name(model_path.name)
)
source_files = sorted(model_path.glob("*.py"))
copied = 0
if cache_root.exists():
    for revision_dir in cache_root.iterdir():
        if not revision_dir.is_dir() or revision_dir.name == "__pycache__":
            continue
        for src in source_files:
            dst = revision_dir / src.name
            if not dst.exists() or not filecmp.cmp(src, dst, shallow=False):
                shutil.copyfile(src, dst)
                copied += 1
    if copied:
        importlib.invalidate_caches()

for ref in sorted(set(class_refs)):
    get_class_from_dynamic_module(ref, str(model_path), local_files_only=True)

print(f"prewarm_remote_code_cache=ok copied_py_files:{copied}")
PY
}

prewarm_remote_code_cache

(
  cd "${VLMEVAL_DIR}"
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
  export CUDA_VISIBLE_DEVICES="${GPUS}"
  export NO_PROXY="${NO_PROXY_VALUE}"
  export no_proxy="${NO_PROXY_VALUE}"
  export USE_COT="${USE_COT}"
  export PRED_FORMAT=xlsx
  unset SPLIT_THINK
  "${CMD[@]}"
) >"${LOG_FILE}" 2>&1 || {
  status=$?
  echo "VLMEvalKit run failed with status ${status}. Tail of log:" >&2
  tail -n 120 "${LOG_FILE}" >&2 || true
  exit "${status}"
}

echo "VLMEvalKit run finished."

if [[ "${DO_MERGE}" -eq 1 ]]; then
  MERGE_ARGS=(
    "${REPO_ROOT}/scripts/merge_vlmeval_results_to_mmr.py"
    --csv "${CSV_PATH}"
    --run-root "${WORK_DIR}"
    --model "${MODEL}"
    --model-path "${MODEL_PATH}"
    --judge "${JUDGE_MODEL}"
    --fix-mistral-regex
  )
  if [[ "${DO_BACKUP}" -eq 1 ]]; then
    MERGE_ARGS+=(--backup)
  fi
  PYTHONPATH="${VLMEVAL_DIR}:${PYTHONPATH:-}" "${CONDA_ENV}/bin/python" "${MERGE_ARGS[@]}" | tee -a "${LOG_FILE}"
else
  echo "Skipping merge_to_mmr_csv because benchmarks are not exactly the standard 7 or --no-merge was set." | tee -a "${LOG_FILE}"
fi

"${CONDA_ENV}/bin/python" - "$WORK_DIR" "$MODEL" <<'PY' | tee -a "${LOG_FILE}"
from pathlib import Path
import pandas as pd
import sys

work_dir = Path(sys.argv[1])
model = sys.argv[2]
model_root = work_dir / model
tdirs = sorted(model_root.glob("T*"), key=lambda p: p.stat().st_mtime)
if not tdirs:
    raise SystemExit(f"No run directory found under {model_root}")
run = tdirs[-1]
print(f"latest_run_dir={run}")
metric_paths = sorted(set(run.glob("*_score.csv")) | set(run.glob("*_acc.csv")))
for path in metric_paths:
    try:
        df = pd.read_csv(path)
        if "Overall" in df.columns:
            value = df["Overall"].iloc[0]
        elif "accuracy" in df.columns:
            value = df["accuracy"].iloc[0]
        elif "acc" in df.columns:
            value = df["acc"].iloc[0]
        else:
            value = "-"
        print(f"metric_csv={path.name} overall={value}")
    except Exception as exc:
        print(f"metric_csv={path.name} read_error={exc}")
PY

echo "done"
