#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv}"
UV_BIN="${UV_BIN:-$HOME/.local/bin/uv}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache-pr14}"
PER_INSTANCE_TIME_LIMIT="${PER_INSTANCE_TIME_LIMIT:-300}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/minlptests-full-benchmark}"
JULIA_BIN="${JULIA_BIN:-julia}"
JULIA_CHANNEL="${JULIA_CHANNEL:-+release}"
FORCE_RERUN="${FORCE_RERUN:-0}"
RUN_DISCOPT="${RUN_DISCOPT:-1}"
RUN_ALPINE="${RUN_ALPINE:-1}"
RUN_COMPARISON="${RUN_COMPARISON:-1}"
ALPINE_MIP_SOLVER="${ALPINE_MIP_SOLVER:-highs}"
JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"
DISCOPT_INPUT_JSON="${DISCOPT_INPUT_JSON:-}"
PYTHON_RUNNER_LABEL=""
declare -a PYTHON_RUNNER=()

ALPINE_PROJECT="${ALPINE_PROJECT:-${REPO_ROOT}/../Alpine.jl}"
MINLPTESTS_PATH="${MINLPTESTS_PATH:-${REPO_ROOT}/../MINLPTests.jl}"

DISCOPT_JSON="${OUTPUT_DIR}/minlptests-full-discopt.json"
DISCOPT_MD="${OUTPUT_DIR}/minlptests-full-discopt.md"
ALPINE_REQUEST="${OUTPUT_DIR}/alpine-full-request.tsv"
ALPINE_JSONL="${OUTPUT_DIR}/alpine-full-results.jsonl"
COMPARISON_JSON="${OUTPUT_DIR}/minlptests-full-comparison.json"
COMPARISON_MD="${OUTPUT_DIR}/minlptests-full-comparison.md"
RUN_LOG="${OUTPUT_DIR}/run.log"

mkdir -p "${OUTPUT_DIR}"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

log() {
  echo "[$(timestamp)] $*"
}

resolve_python_runner() {
  if [[ -x "${VENV_DIR}/bin/python" ]]; then
    PYTHON_RUNNER=("${VENV_DIR}/bin/python")
    PYTHON_RUNNER_LABEL="${VENV_DIR}/bin/python"
    return 0
  fi

  if [[ -x "${UV_BIN}" ]]; then
    PYTHON_RUNNER=("${UV_BIN}" "run" "--extra" "dev" "python")
    PYTHON_RUNNER_LABEL="${UV_BIN} run --extra dev python"
    return 0
  fi

  echo "Missing Python runner: neither ${VENV_DIR}/bin/python nor ${UV_BIN} is executable" >&2
  exit 1
}

if [[ "${NO_INTERNAL_LOG:-0}" != "1" ]]; then
  exec > >(tee -a "${RUN_LOG}") 2>&1
fi

trap 'log "ERROR: benchmark runner failed at line ${LINENO}"' ERR

log "Repository: ${REPO_ROOT}"
log "Alpine.jl: ${ALPINE_PROJECT}"
log "MINLPTests.jl: ${MINLPTESTS_PATH}"
log "Output dir: ${OUTPUT_DIR}"
log "Per-instance time limit: ${PER_INSTANCE_TIME_LIMIT}s"
log "Force rerun: ${FORCE_RERUN}"
log "Run discopt: ${RUN_DISCOPT}"
log "Run Alpine: ${RUN_ALPINE}"
log "Run comparison: ${RUN_COMPARISON}"
log "Alpine MIP solver: ${ALPINE_MIP_SOLVER}"
echo

resolve_python_runner
log "Python runner: ${PYTHON_RUNNER_LABEL}"
log "JAX_PLATFORMS: ${JAX_PLATFORMS}"
log "JAX_ENABLE_X64: ${JAX_ENABLE_X64}"

if [[ ! -d "${ALPINE_PROJECT}" ]]; then
  echo "Missing Alpine.jl checkout: ${ALPINE_PROJECT}" >&2
  echo "Set ALPINE_PROJECT or place Alpine.jl next to this repo." >&2
  exit 1
fi

if [[ ! -d "${MINLPTESTS_PATH}" ]]; then
  echo "Missing MINLPTests.jl checkout: ${MINLPTESTS_PATH}" >&2
  echo "Set MINLPTESTS_PATH or place MINLPTests.jl next to this repo." >&2
  exit 1
fi

COMPARISON_DISCOPT_JSON="${DISCOPT_JSON}"
if [[ -n "${DISCOPT_INPUT_JSON}" ]]; then
  COMPARISON_DISCOPT_JSON="${DISCOPT_INPUT_JSON}"
fi

if [[ "${RUN_DISCOPT}" == "1" ]]; then
  if [[ "${FORCE_RERUN}" != "1" && -s "${DISCOPT_JSON}" && -s "${DISCOPT_MD}" ]]; then
    log "Skipping Step 1/4: discopt outputs already exist"
  else
    log "=== Step 1/4: discopt AMP full translated suite ==="
    (
      cd "${REPO_ROOT}"
      UV_CACHE_DIR="${UV_CACHE_DIR}" PYTHONPATH=python \
        JAX_PLATFORMS="${JAX_PLATFORMS}" JAX_ENABLE_X64="${JAX_ENABLE_X64}" \
        "${PYTHON_RUNNER[@]}" scripts/collect_minlptests_status.py \
          --skip-alpine \
          --include-convex \
          --per-instance-time-limit "${PER_INSTANCE_TIME_LIMIT}" \
          --output-json "${DISCOPT_JSON}" \
          --output-markdown "${DISCOPT_MD}"
    )
    log "discopt outputs:"
    log "  ${DISCOPT_JSON}"
    log "  ${DISCOPT_MD}"
  fi
else
  log "Skipping Step 1/4: discopt run disabled"
fi
echo

if [[ "${RUN_ALPINE}" == "1" ]]; then
  if [[ "${FORCE_RERUN}" != "1" && -s "${ALPINE_REQUEST}" ]]; then
    log "Skipping Step 2/4: Alpine request file already exists"
  else
    log "=== Step 2/4: build Alpine request file ==="
    (
      cd "${REPO_ROOT}"
      UV_CACHE_DIR="${UV_CACHE_DIR}" PYTHONPATH=python \
        JAX_PLATFORMS="${JAX_PLATFORMS}" JAX_ENABLE_X64="${JAX_ENABLE_X64}" \
        "${PYTHON_RUNNER[@]}" - <<PY
from pathlib import Path
from scripts.collect_minlptests_status import case_catalog, load_test_module

mod = load_test_module()
cases = case_catalog(
    mod,
    include_convex=True,
    per_instance_time_limit=float("${PER_INSTANCE_TIME_LIMIT}"),
)
Path("${ALPINE_REQUEST}").write_text(
    "".join(
        f"{case['instance'].problem_id}\t{case['category']}\t{case['symbol']}\n"
        for case in cases
    ),
    encoding="utf-8",
)
print(f"Wrote {len(cases)} cases to ${ALPINE_REQUEST}")
PY
    )
  fi
  echo

  if [[ "${FORCE_RERUN}" != "1" && -s "${ALPINE_JSONL}" ]]; then
    log "Skipping Step 3/4: Alpine output already exists"
  else
    log "=== Step 3/4: Alpine.jl full translated suite ==="
    JULIA_CMD=("${JULIA_BIN}")
    if [[ -n "${JULIA_CHANNEL}" ]]; then
      JULIA_CMD+=("${JULIA_CHANNEL}")
    fi
    JULIA_CMD+=(
      "--project=."
      "${REPO_ROOT}/scripts/alpine_minlptests_status.jl"
      "${ALPINE_REQUEST}"
      "${ALPINE_JSONL}"
      "${MINLPTESTS_PATH}"
      "${PER_INSTANCE_TIME_LIMIT}"
    )
    (
      cd "${ALPINE_PROJECT}"
      ALPINE_MIP_SOLVER="${ALPINE_MIP_SOLVER}" "${JULIA_CMD[@]}"
    )
    log "Alpine output:"
    log "  ${ALPINE_JSONL}"
  fi
else
  log "Skipping Step 2/4 and Step 3/4: Alpine run disabled"
fi
echo

if [[ "${RUN_COMPARISON}" == "1" ]]; then
  if [[ ! -s "${COMPARISON_DISCOPT_JSON}" ]]; then
    echo "Missing discopt JSON for comparison: ${COMPARISON_DISCOPT_JSON}" >&2
    exit 1
  fi
  if [[ ! -s "${ALPINE_JSONL}" ]]; then
    echo "Missing Alpine JSONL for comparison: ${ALPINE_JSONL}" >&2
    exit 1
  fi

  if [[ "${FORCE_RERUN}" != "1" && -s "${COMPARISON_JSON}" && -s "${COMPARISON_MD}" ]]; then
    log "Skipping Step 4/4: comparison outputs already exist"
  else
    log "=== Step 4/4: synthesize combined comparison ==="
    (
      cd "${REPO_ROOT}"
      UV_CACHE_DIR="${UV_CACHE_DIR}" PYTHONPATH=python \
        JAX_PLATFORMS="${JAX_PLATFORMS}" JAX_ENABLE_X64="${JAX_ENABLE_X64}" \
        "${PYTHON_RUNNER[@]}" - <<PY
import json
from pathlib import Path

from scripts.collect_minlptests_status import build_markdown, compare_outcomes, summarize

output_dir = Path("${OUTPUT_DIR}")
discopt_payload = json.loads(Path("${COMPARISON_DISCOPT_JSON}").read_text(encoding="utf-8"))
discopt_records = discopt_payload["discopt"]
alpine_records = [
    json.loads(line)
    for line in Path("${ALPINE_JSONL}").read_text(encoding="utf-8").splitlines()
    if line.strip()
]

payload = {
    "discopt": discopt_records,
    "alpine": alpine_records,
    "discopt_summary": summarize(discopt_records),
    "alpine_summary": summarize(alpine_records),
    "comparison_summary": compare_outcomes(discopt_records, alpine_records),
}

Path("${COMPARISON_JSON}").write_text(
    json.dumps(payload, indent=2, sort_keys=True),
    encoding="utf-8",
)
Path("${COMPARISON_MD}").write_text(
    build_markdown(discopt_records, alpine_records),
    encoding="utf-8",
)
print("Comparison summary:", payload["comparison_summary"])
print("Wrote ${COMPARISON_JSON}")
print("Wrote ${COMPARISON_MD}")
PY
    )
  fi
else
  log "Skipping Step 4/4: comparison disabled"
fi

echo
log "Full benchmark complete."
log "Artifacts:"
if [[ "${RUN_DISCOPT}" == "1" ]]; then
  log "  ${DISCOPT_JSON}"
  log "  ${DISCOPT_MD}"
else
  log "  discopt input: ${COMPARISON_DISCOPT_JSON}"
fi
log "  ${ALPINE_JSONL}"
log "  ${COMPARISON_JSON}"
log "  ${COMPARISON_MD}"
log "  ${RUN_LOG}"
