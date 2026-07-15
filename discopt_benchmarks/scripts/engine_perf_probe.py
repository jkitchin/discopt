#!/usr/bin/env python
"""EP0 engine-performance probe (issue #632).

A reusable, in-container measurement instrument for the EP series
(``docs/dev/engine-performance-plan.md``). It reports, per named instance, the
per-node relaxation-construction costs the EP items attack — with **no
solver-math change and no new flag** — so every later item (EP1..EP6) has a
comparable before/after column.

Per instance it measures:

* **ctor** — wall time to build the reusable relaxer
  (:class:`~discopt._jax.mccormick_lp.MccormickLPRelaxer`); this pays the fixed
  first-JAX-trace + incremental double-build setup cost.
* **root** — wall time of the first ``solve_at_node`` over the full model box.
* **ms/node** — mean wall time of ``solve_at_node`` over ``--children N``
  simulated child boxes (each shrinks one variable's box to its lower half, the
  branching pattern used in the 2026-07-13 in-container profile). The root solve
  runs first so these children measure the *warm* steady-state per-node cost.
* **builds** — the number of ``build_milp_relaxation`` calls for ONE end-to-end
  ``model.solve()``, counted by an in-process monkeypatched wrapper installed in
  the probe's own process (library code is NOT instrumented).

``--profile`` additionally prints a cProfile top-20 (cumulative) of one
``model.solve()``.

Usage (from repo root, extension built, venv active)::

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
        python discopt_benchmarks/scripts/engine_perf_probe.py
    ... engine_perf_probe.py --instances nvs09,ex1226 --children 20 --json out.json
    ... engine_perf_probe.py --profile

Determinism: no randomness, no timestamps; child boxes are a fixed bisection
schedule. Timings are wall-clock and will vary run-to-run — record them as the
order-of-magnitude baseline, not to the microsecond.
"""

from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import sys
import time
from contextlib import contextmanager
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator

_REPO = Path(__file__).resolve().parents[2]
_NL_DIR = _REPO / "python" / "tests" / "data" / "minlplib_nl"
sys.path.insert(0, str(_REPO / "python"))

_DEFAULT_INSTANCES = ("nvs09", "ex1226", "bchoco06")


# ---------------------------------------------------------------------------
# build_milp_relaxation call counter (in-process monkeypatch, no library edit)
# ---------------------------------------------------------------------------
@contextmanager
def _count_builds() -> Iterator[list[int]]:
    """Count ``build_milp_relaxation`` invocations during the ``with`` body.

    Wraps the function at its source module AND at every already-imported module
    that bound the name at import time (e.g. ``mccormick_lp`` does
    ``from ...milp_relaxation import build_milp_relaxation`` at module scope, so
    patching only the source module would miss its calls). Callers that import
    the name *inside* a function re-fetch it from the source module at call time
    and so are covered by the source-module patch. All bindings are restored on
    exit. Pure instrumentation, confined to this process.
    """
    from discopt._jax import milp_relaxation as _mr

    orig = _mr.build_milp_relaxation
    counter = [0]

    def wrapper(*args, **kwargs):
        counter[0] += 1
        return orig(*args, **kwargs)

    patched: list[object] = []
    _mr.build_milp_relaxation = wrapper  # type: ignore[assignment]
    patched.append(_mr)
    for mod in list(sys.modules.values()):
        if mod is _mr or mod is None:
            continue
        if getattr(mod, "build_milp_relaxation", None) is orig:
            mod.build_milp_relaxation = wrapper  # type: ignore[attr-defined]
            patched.append(mod)
    try:
        yield counter
    finally:
        for mod in patched:
            mod.build_milp_relaxation = orig  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Child-box schedule
# ---------------------------------------------------------------------------
def _child_boxes(
    lb: np.ndarray, ub: np.ndarray, n: int
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """Yield ``n`` child boxes, each shrinking one variable's box to its lower half.

    Deterministic: child ``i`` bisects finite-width column ``i mod k`` (skipping
    already-degenerate columns). Mirrors the spatial-branch pattern that the
    per-node profile in engine-performance-plan.md §1 uses.
    """
    width = ub - lb
    branchable = [c for c in range(lb.size) if np.isfinite(width[c]) and width[c] > 1e-9]
    if not branchable:
        branchable = list(range(lb.size))
    children: list[tuple[int, np.ndarray, np.ndarray]] = []
    for i in range(n):
        col = branchable[i % len(branchable)]
        clb = lb.copy()
        cub = ub.copy()
        mid = 0.5 * (clb[col] + cub[col])
        cub[col] = mid  # keep the lower half
        children.append((col, clb, cub))
    return children


# ---------------------------------------------------------------------------
# Per-instance probe
# ---------------------------------------------------------------------------
def probe_instance(
    name: str,
    *,
    children: int,
    solve_time_limit: float,
    do_profile: bool,
) -> dict:
    from discopt._jax.mccormick_lp import MccormickLPRelaxer
    from discopt._jax.model_utils import flat_variable_bounds
    from discopt.modeling.core import from_nl

    nl_path = _NL_DIR / f"{name}.nl"
    if not nl_path.exists():
        return {"instance": name, "error": f"missing {nl_path}"}

    model = from_nl(str(nl_path))
    lb, ub = flat_variable_bounds(model)
    n_cols = int(lb.size)

    # 1) relaxer construction time
    t0 = time.perf_counter()
    relaxer = MccormickLPRelaxer(model)
    ctor_s = time.perf_counter() - t0

    # 2) root solve_at_node time (also warms JAX/first-build caches)
    t0 = time.perf_counter()
    root_res = relaxer.solve_at_node(lb.copy(), ub.copy())
    root_s = time.perf_counter() - t0
    root_bound = (
        float(root_res.lower_bound) if getattr(root_res, "lower_bound", None) is not None else None
    )

    # 3) ms/node over simulated child boxes (warm, steady state)
    child_times: list[float] = []
    for _col, clb, cub in _child_boxes(lb, ub, children):
        t0 = time.perf_counter()
        relaxer.solve_at_node(clb, cub)
        child_times.append(time.perf_counter() - t0)
    ms_per_node = 1000.0 * float(np.mean(child_times)) if child_times else None
    ms_per_node_min = 1000.0 * float(np.min(child_times)) if child_times else None

    # 4) build_milp_relaxation call count for one end-to-end solve
    solve_model = from_nl(str(nl_path))
    with _count_builds() as counter:
        t0 = time.perf_counter()
        result = solve_model.solve(time_limit=solve_time_limit)
        solve_s = time.perf_counter() - t0
    build_calls = counter[0]

    row = {
        "instance": name,
        "n_cols": n_cols,
        "ctor_s": ctor_s,
        "root_s": root_s,
        "root_status": root_res.status,
        "root_bound": root_bound,
        "ms_per_node": ms_per_node,
        "ms_per_node_min": ms_per_node_min,
        "children": children,
        "build_calls": build_calls,
        "solve_s": solve_s,
        "solve_status": getattr(result, "status", None),
        "solve_objective": getattr(result, "objective", None),
        "solve_node_count": getattr(result, "node_count", None),
    }

    if do_profile:
        row["profile_top20"] = _profile_solve(str(nl_path), solve_time_limit)
    return row


def _profile_solve(nl_path: str, solve_time_limit: float) -> str:
    from discopt.modeling.core import from_nl

    model = from_nl(nl_path)
    prof = cProfile.Profile()
    prof.enable()
    model.solve(time_limit=solve_time_limit)
    prof.disable()
    buf = StringIO()
    stats = pstats.Stats(prof, stream=buf).sort_stats("cumulative")
    stats.print_stats(20)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _fmt(x: float | None, width: int = 10, prec: int = 3) -> str:
    if x is None:
        return "None".rjust(width)
    return f"{x:{width}.{prec}f}"


def print_table(rows: list[dict]) -> None:
    print("\n=== EP0 engine-performance probe ===")
    hdr = (
        f"{'instance':<14}{'cols':>6}{'ctor_s':>10}{'root_s':>10}"
        f"{'ms/node':>10}{'builds':>8}{'nodes':>8}{'obj':>14}  status"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if "error" in r:
            print(f"{r['instance']:<14}  ERROR: {r['error']}")
            continue
        print(
            f"{r['instance']:<14}{r['n_cols']:>6}{_fmt(r['ctor_s'])}{_fmt(r['root_s'])}"
            f"{_fmt(r['ms_per_node'])}{r['build_calls']:>8}"
            f"{(r['solve_node_count'] if r['solve_node_count'] is not None else 0):>8}"
            f"{_fmt(r['solve_objective'], 14, 5)}  {r['solve_status']}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--instances",
        type=str,
        default=",".join(_DEFAULT_INSTANCES),
        help="comma-separated instance stems under python/tests/data/minlplib_nl",
    )
    ap.add_argument(
        "--children", type=int, default=10, help="number of simulated child boxes (default 10)"
    )
    ap.add_argument(
        "--solve-time-limit",
        type=float,
        default=120.0,
        help="time limit (s) for the end-to-end solve that counts builds",
    )
    ap.add_argument("--profile", action="store_true", help="also print cProfile top-20 cumulative")
    ap.add_argument("--json", type=str, default=None, help="write full results to this JSON path")
    args = ap.parse_args()

    instances = [s.strip() for s in args.instances.split(",") if s.strip()]
    rows: list[dict] = []
    for name in instances:
        rows.append(
            probe_instance(
                name,
                children=args.children,
                solve_time_limit=args.solve_time_limit,
                do_profile=args.profile,
            )
        )

    print_table(rows)

    if args.profile:
        for r in rows:
            if "profile_top20" in r:
                print(f"\n--- cProfile top-20 (cumulative): {r['instance']} ---")
                print(r["profile_top20"])

    if args.json:
        Path(args.json).write_text(json.dumps({"rows": rows}, indent=2, sort_keys=True))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
