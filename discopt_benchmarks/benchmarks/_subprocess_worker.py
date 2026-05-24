"""Subprocess worker that solves one MINLP instance with discopt.

Invoked by :mod:`discopt_benchmarks.benchmarks.scaled_runner` as

    python -m benchmarks._subprocess_worker \\
        --instance <name> --nl-path <path> \\
        --time-limit <s> --mem-limit-mb <mb> \\
        --options-json '{...}' --out-json <path>

Writes a single ``SolveResult.to_dict()`` JSON object to ``--out-json`` on
success, or a JSON ``{"_error": "...", "_status": "error"}`` on failure.

Memory cap is enforced via ``resource.setrlimit(RLIMIT_AS, ...)`` so the
parent process can fail-fast on OOM without dragging the whole sweep down.
Wall-clock cap is enforced by the parent via ``Popen.wait(timeout=...)``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path


def _install_memory_limit(mb: int) -> None:
    if mb <= 0:
        return
    try:
        import resource

        bytes_ = mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (bytes_, bytes_))
    except Exception as e:  # noqa: BLE001
        print(f"[worker] WARN: could not set RLIMIT_AS={mb}MB: {e}", file=sys.stderr)


def _solve(
    instance: str,
    nl_path: str,
    time_limit: float,
    options: dict,
) -> dict:
    """Run model.solve and return a SolveResult-shaped dict."""
    from benchmarks.metrics import SolveResult, SolveStatus
    import discopt.modeling as dm

    start = time.monotonic()
    try:
        model = dm.from_nl(nl_path)
    except Exception as e:  # noqa: BLE001
        return SolveResult(
            instance=instance,
            solver=options.get("_solver_name", "discopt"),
            status=SolveStatus.ERROR,
            wall_time=time.monotonic() - start,
        ).to_dict() | {"_error": f"from_nl failed: {e}"}

    opts = dict(options)
    opts.pop("_solver_name", None)
    gap_tol = opts.pop("gap_tolerance", 1e-4)
    max_nodes = opts.pop("max_nodes", 100_000)
    opts.pop("gpu", None)

    try:
        result = model.solve(
            time_limit=time_limit,
            gap_tolerance=gap_tol,
            max_nodes=max_nodes,
            **opts,
        )
    except Exception as e:  # noqa: BLE001
        return SolveResult(
            instance=instance,
            solver=options.get("_solver_name", "discopt"),
            status=SolveStatus.ERROR,
            wall_time=time.monotonic() - start,
        ).to_dict() | {"_error": f"solve raised: {e}", "_traceback": traceback.format_exc()}

    status_map = {
        "optimal": SolveStatus.OPTIMAL,
        "feasible": SolveStatus.FEASIBLE,
        "infeasible": SolveStatus.INFEASIBLE,
        "time_limit": SolveStatus.TIME_LIMIT,
        "node_limit": SolveStatus.TIME_LIMIT,
    }
    bench_status = status_map.get(getattr(result, "status", "unknown"), SolveStatus.UNKNOWN)

    wt = result.wall_time if getattr(result, "wall_time", 0) > 0 else 1e-10
    rust_frac = (result.rust_time / wt) if getattr(result, "rust_time", None) else None
    jax_frac = (result.jax_time / wt) if getattr(result, "jax_time", None) else None
    py_frac = (result.python_time / wt) if getattr(result, "python_time", None) else None

    return SolveResult(
        instance=instance,
        solver=options.get("_solver_name", "discopt"),
        status=bench_status,
        objective=getattr(result, "objective", None),
        bound=getattr(result, "bound", None),
        wall_time=float(getattr(result, "wall_time", time.monotonic() - start)),
        node_count=int(getattr(result, "node_count", 0) or 0),
        rust_time_fraction=rust_frac,
        jax_time_fraction=jax_frac,
        python_time_fraction=py_frac,
    ).to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--nl-path", required=True)
    parser.add_argument("--time-limit", type=float, required=True)
    parser.add_argument("--mem-limit-mb", type=int, default=0,
                        help="Address-space cap in MB (0 disables)")
    parser.add_argument("--options-json", type=str, default="{}",
                        help="JSON-encoded solver options dict")
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()

    _install_memory_limit(args.mem_limit_mb)

    try:
        options = json.loads(args.options_json) if args.options_json else {}
    except json.JSONDecodeError as e:
        out = {"_error": f"bad options-json: {e}", "_status": "error", "instance": args.instance}
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(out))
        sys.exit(2)

    nl_path = Path(args.nl_path)
    if not nl_path.exists():
        out = {"_error": f"nl file missing: {nl_path}", "status": "error", "instance": args.instance}
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(out))
        sys.exit(3)

    payload = _solve(args.instance, str(nl_path), args.time_limit, options)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, default=str))


if __name__ == "__main__":
    main()
