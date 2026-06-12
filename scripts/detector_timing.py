"""Detector-timing benchmark — baseline the Rust-port discussion for issue #38.

discopt's structure detectors in ``python/discopt/_jax/convexity/`` and the new
``python/discopt/_jax/monotonicity.py`` are pure Python today. A Rust port only
pays off if (a) detection is a measurable share of work and (b) the hot path is
*interpreter-bound* DAG walking rather than already-vectorised numpy/JAX kernels.
This script measures both, on two workloads, and prints a cProfile breakdown so
the bottleneck is visible rather than assumed.

Paths timed (the public detector entry points):
  * ``classify_expr``         — lattice curvature walk (DCP composition rules)
  * ``certify_convex``        — interval-AD Hessian + eigenvalue lower bound
  * ``classify_monotonicity`` — forward interval-gradient orthant test
  * ``evaluate_interval``     — forward natural-range interval enclosure

Workloads:
  1. Corpus      — the 40-instance suspect_oracle corpus (realistic, mixed atoms).
  2. Scaling     — synthetic convex expressions of growing node count N, to expose
                   per-node interpreter cost and how each detector scales in N.

Usage::

    python scripts/detector_timing.py            # full run + cProfile
    python scripts/detector_timing.py --quick     # fewer repeats / smaller sweep
    python scripts/detector_timing.py --no-profile
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import sys
import time
from pathlib import Path
from typing import Callable

import discopt.modeling as dm
from discopt import Model
from discopt._jax.convexity import certify_convex, classify_expr
from discopt._jax.convexity.interval_eval import evaluate_interval
from discopt._jax.monotonicity import classify_monotonicity

_ORACLE_DIR = Path(__file__).resolve().parent / "suspect_oracle"
if str(_ORACLE_DIR) not in sys.path:
    sys.path.insert(0, str(_ORACLE_DIR))


# --------------------------------------------------------------------------- #
# Timing helpers
# --------------------------------------------------------------------------- #
def _time(fn: Callable[[], object], repeat: int) -> float:
    """Best-of-``repeat`` median wall time (seconds) for a zero-arg call."""
    samples = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    samples.sort()
    return samples[len(samples) // 2]


def _count_nodes(expr: object, seen: set | None = None) -> int:
    """Count distinct Expression nodes reachable from ``expr`` (DAG-aware)."""
    from discopt.modeling.core import Expression

    if seen is None:
        seen = set()
    if id(expr) in seen:
        return 0
    seen.add(id(expr))
    n = 1
    for v in vars(expr).values():
        if isinstance(v, Expression):
            n += _count_nodes(v, seen)
        elif isinstance(v, (list, tuple)):
            for it in v:
                if isinstance(it, Expression):
                    n += _count_nodes(it, seen)
    return n


# --------------------------------------------------------------------------- #
# The four detector paths, each as a () -> value thunk over (expr, model)
# --------------------------------------------------------------------------- #
DETECTORS: dict[str, Callable[[object, Model], Callable[[], object]]] = {
    "classify_expr": lambda e, m: (lambda: classify_expr(e, m)),
    "certify_convex": lambda e, m: (lambda: certify_convex(e, m)),
    "classify_monotonicity": lambda e, m: (lambda: classify_monotonicity(e, m)),
    "evaluate_interval": lambda e, m: (lambda: evaluate_interval(e, m)),
}


# --------------------------------------------------------------------------- #
# Workload 1: the suspect_oracle corpus
# --------------------------------------------------------------------------- #
def _corpus_items() -> list[tuple[str, object, Model, int]]:
    """`(key, raw_body, model, node_count)` for every corpus item."""
    from corpus import INSTANCES  # noqa: PLC0415
    from render_discopt import build_discopt_items  # noqa: PLC0415

    out = []
    for inst in INSTANCES:
        model, items = build_discopt_items(inst)
        for item in items:
            body = item["body"]
            out.append(
                (f"{inst['name']}::{item['key']}", body, model, _count_nodes(body))
            )
    return out


def run_corpus(repeat: int) -> None:
    items = _corpus_items()
    total_nodes = sum(n for _, _, _, n in items)
    print(f"\n=== Workload 1: suspect_oracle corpus "
          f"({len(items)} items, {total_nodes} total nodes) ===")
    print(f"{'detector':<24}{'total ms':>12}{'µs / item':>14}{'µs / node':>14}")
    for name, make in DETECTORS.items():
        total = 0.0
        for _key, body, model, _n in items:
            try:
                total += _time(make(body, model), repeat)
            except Exception:  # a detector may not support every atom; skip
                pass
        per_item = total / len(items) * 1e6
        per_node = total / total_nodes * 1e6
        print(f"{name:<24}{total * 1e3:>12.3f}{per_item:>14.2f}{per_node:>14.3f}")


# --------------------------------------------------------------------------- #
# Workload 2: synthetic scaling sweep
# --------------------------------------------------------------------------- #
def _build_chain(n_terms: int) -> tuple[object, Model]:
    """A convex expression with ~``n_terms`` atom terms: sum of squares, exp,
    softplus-ish logs, over distinct variables — every term a real atom so the
    detectors do genuine per-node work (not a trivially-foldable constant)."""
    m = Model(f"scale{n_terms}")
    xs = [m.continuous(f"x{i}", lb=0.1, ub=2.0) for i in range(n_terms)]
    expr: object = dm.exp(xs[0])
    for i, x in enumerate(xs):
        if i % 3 == 0:
            expr = expr + x * x
        elif i % 3 == 1:
            expr = expr + dm.exp(x)
        else:
            expr = expr - dm.log(x)  # -log is convex
    return expr, m


def run_scaling(sizes: list[int], repeat: int) -> None:
    print("\n=== Workload 2: synthetic scaling sweep (convex sum) ===")
    header = f"{'N nodes':>10}" + "".join(f"{d:>22}" for d in DETECTORS)
    print(header)
    print(f"{'':>10}" + "".join(f"{'ms (µs/node)':>22}" for _ in DETECTORS))
    for n_terms in sizes:
        expr, model = _build_chain(n_terms)
        nodes = _count_nodes(expr)
        row = f"{nodes:>10}"
        for _name, make in DETECTORS.items():
            try:
                t = _time(make(expr, model), repeat)
                row += f"{t * 1e3:>11.2f}({t / nodes * 1e6:>6.2f})"
            except Exception:
                row += f"{'err':>22}"
        print(row)


# --------------------------------------------------------------------------- #
# cProfile breakdown — the Rust-port signal
# --------------------------------------------------------------------------- #
def run_profile(n_terms: int, repeat: int) -> None:
    expr, model = _build_chain(n_terms)
    nodes = _count_nodes(expr)
    print(f"\n=== cProfile: certify_convex on N={nodes}-node expr "
          f"(×{repeat}) — the likely hot path ===")
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(repeat):
        certify_convex(expr, model)
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(20)
    print(s.getvalue())


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="fewer repeats / smaller sweep")
    ap.add_argument("--no-profile", action="store_true", help="skip the cProfile breakdown")
    args = ap.parse_args(argv)

    repeat = 3 if args.quick else 7
    sizes = [4, 16, 64] if args.quick else [4, 16, 64, 128, 256]

    run_corpus(repeat=repeat)
    run_scaling(sizes=sizes, repeat=repeat)
    if not args.no_profile:
        run_profile(n_terms=sizes[-1], repeat=20 if args.quick else 50)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
