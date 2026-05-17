#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -eq 0 ]]; then
    echo "usage: $0 <pytest-command> [pytest-args...]" >&2
    exit 2
fi

memory_mb="${PYTEST_MEMORY_LIMIT_MB:-16384}"
cpu_seconds="${PYTEST_CPU_LIMIT_SECONDS:-0}"
dry_run="${RUN_MEMORY_CAPPED_PYTEST_DRY_RUN:-0}"

cmd=("$@")
limits=()

export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.60}"

if [[ "$memory_mb" != "0" ]]; then
    memory_bytes=$((memory_mb * 1024 * 1024))
    limits+=("--as=${memory_bytes}")
fi

if [[ "$cpu_seconds" != "0" ]]; then
    limits+=("--cpu=${cpu_seconds}")
fi

if command -v prlimit >/dev/null 2>&1 && [[ "${#limits[@]}" -gt 0 ]]; then
    echo "==> pytest resource limits: memory=${memory_mb}MB cpu=${cpu_seconds}s"
    echo "==> JAX allocator: preallocate=${XLA_PYTHON_CLIENT_PREALLOCATE} mem_fraction=${XLA_PYTHON_CLIENT_MEM_FRACTION}"
    if [[ "$dry_run" == "1" ]]; then
        printf 'prlimit'
        printf ' %q' "${limits[@]}"
        printf ' --'
        printf ' %q' "${cmd[@]}"
        printf '\n'
        exit 0
    fi
    exec prlimit "${limits[@]}" -- "${cmd[@]}"
fi

if [[ "$dry_run" == "1" ]]; then
    printf '%q' "${cmd[0]}"
    printf ' %q' "${cmd[@]:1}"
    printf '\n'
    exit 0
fi

if [[ "${#limits[@]}" -gt 0 ]]; then
    echo "WARNING: prlimit is unavailable; requested pytest resource caps are NOT enforced (memory=${memory_mb}MB cpu=${cpu_seconds}s)." >&2
    echo "WARNING: this commonly happens on macOS; run in Linux/CI when memory-capped behavior matters." >&2
fi
exec "${cmd[@]}"
