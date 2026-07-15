"""G1 entry experiment: callback census + fused-prototype probe on nvs05.

Run with:
    PYTHONPATH=<worktree>/python python docs/dev/data/g1/g1_entry_experiment.py

Two parts (per baron-gap-plan.md §3):
  1. Callback census on ONE nvs05 solve (time_limit=20): per-quantity
     {calls, total us, us/call}, splitting jit+compute vs host-conversion.
  2. Throwaway fused prototype: one jitted (f, g, c, J) pytree, iterate memo
     keyed on x.tobytes(), jax.device_get once per iterate. Measure the same
     probe (callback-overhead only, replaying the census-captured iterates).

Kill criterion: fused/memoized < 2x callback-overhead reduction, or any
bound/node deviates.
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict

import numpy as np

NL_PATH = "python/tests/data/minlplib_nl/nvs05.nl"
TIME_LIMIT = 20.0


def _census_solve():
    """Instrument NLPEvaluator methods, run one nvs05 solve, return census."""
    import discopt._jax.nlp_evaluator as nev
    from discopt.modeling.core import from_nl

    Ev = nev.NLPEvaluator

    counters: dict[str, dict[str, float]] = defaultdict(
        lambda: {"calls": 0, "total_ns": 0, "convert_ns": 0}
    )
    # Record (x, quantity) to see which quantities share an iterate.
    iterate_log: list[tuple[bytes, str]] = []

    orig = {
        "objective": Ev.evaluate_objective,
        "gradient": Ev.evaluate_gradient,
        "constraints": Ev.evaluate_constraints,
        "jacobian_values": Ev.evaluate_jacobian_values,
        "jacobian_dense": Ev.evaluate_jacobian,
        "hessian_values": Ev.evaluate_hessian_values,
        "lagrangian_hessian": Ev.evaluate_lagrangian_hessian,
    }

    def _log_x(x, q):
        try:
            iterate_log.append((np.asarray(x, dtype=np.float64).tobytes(), q))
        except Exception:
            pass

    def wrap(name, fn, xarg_is_first=True):
        def inner(self, *a, **k):
            if xarg_is_first and a:
                _log_x(a[0], name)
            t0 = time.perf_counter_ns()
            r = fn(self, *a, **k)
            t1 = time.perf_counter_ns()
            c = counters[name]
            c["calls"] += 1
            c["total_ns"] += t1 - t0
            return r

        return inner

    Ev.evaluate_objective = wrap("objective", orig["objective"])
    Ev.evaluate_gradient = wrap("gradient", orig["gradient"])
    Ev.evaluate_constraints = wrap("constraints", orig["constraints"])
    Ev.evaluate_jacobian_values = wrap("jacobian_values", orig["jacobian_values"])
    Ev.evaluate_jacobian = wrap("jacobian_dense", orig["jacobian_dense"])
    Ev.evaluate_hessian_values = wrap("hessian_values", orig["hessian_values"])
    Ev.evaluate_lagrangian_hessian = wrap("lagrangian_hessian", orig["lagrangian_hessian"])

    model = from_nl(NL_PATH)
    t0 = time.perf_counter()
    res = model.solve(time_limit=TIME_LIMIT, threads=1)
    wall = time.perf_counter() - t0

    # restore
    for k, v in orig.items():
        setattr(Ev, f"evaluate_{k}" if not k.startswith("jacobian_d") else "evaluate_jacobian", v)
    Ev.evaluate_objective = orig["objective"]
    Ev.evaluate_gradient = orig["gradient"]
    Ev.evaluate_constraints = orig["constraints"]
    Ev.evaluate_jacobian_values = orig["jacobian_values"]
    Ev.evaluate_jacobian = orig["jacobian_dense"]
    Ev.evaluate_hessian_values = orig["hessian_values"]
    Ev.evaluate_lagrangian_hessian = orig["lagrangian_hessian"]

    census = {
        name: {
            "calls": c["calls"],
            "total_us": c["total_ns"] / 1e3,
            "us_per_call": (c["total_ns"] / c["calls"] / 1e3) if c["calls"] else 0.0,
        }
        for name, c in counters.items()
    }

    # iterate-sharing analysis: how many distinct x, and how many quantities per x
    per_x = defaultdict(set)
    for xb, q in iterate_log:
        per_x[xb].add(q)
    n_distinct = len(per_x)
    quant_hist = defaultdict(int)
    for qs in per_x.values():
        quant_hist[frozenset(qs)] += 1
    combo_summary = {
        "+".join(sorted(k)): v
        for k, v in sorted(quant_hist.items(), key=lambda kv: -kv[1])
    }

    return {
        "wall_s": wall,
        "objective": _res_field(res, "objective"),
        "bound": _res_field(res, "bound"),
        "node_count": _res_field(res, "node_count"),
        "status": str(_res_field(res, "status")),
        "census": census,
        "n_distinct_iterates": n_distinct,
        "n_total_callbacks": len(iterate_log),
        "iterate_quantity_combos": combo_summary,
    }


def _res_field(res, name):
    for n in (name, f"_{name}"):
        if hasattr(res, n):
            v = getattr(res, n)
            try:
                return float(v)
            except (TypeError, ValueError):
                return v
    return None


def _fused_prototype_probe():
    """Build a fused (f,g,c,J) jit for the nvs05 evaluator; replay a batch of
    iterates through (a) the current per-quantity path and (b) a memoized fused
    path. Measure callback-overhead wall for identical work + verify values match.
    """
    import jax
    import jax.numpy as jnp

    import discopt._jax.nlp_evaluator as nev
    from discopt.modeling.core import from_nl

    model = from_nl(NL_PATH)
    ev = nev.cached_evaluator(model)
    n = ev.n_variables

    # Force structure caches so has_sparse_structure / COO are warm.
    ev._ensure_coo_cache()
    use_sparse = ev.has_sparse_structure()

    # Build the fused jitted function: (f, g, c, J_dense) at (x, params).
    obj_fn = ev._obj_fn_jit
    grad_fn = ev._grad_fn_jit
    cons_fn = ev._cons_fn_jit
    jac_fn = ev._jac_fn_jit

    def fused(x, params):
        f = obj_fn(x, params)
        g = grad_fn(x, params)
        c = cons_fn(x, params) if cons_fn is not None else jnp.zeros(0)
        J = jac_fn(x, params) if jac_fn is not None else jnp.zeros((0, n))
        return f, g, c, J

    fused_jit = jax.jit(fused)

    rng = np.random.default_rng(7)
    lb, ub = ev.variable_bounds
    lb_c = np.where(np.isfinite(lb), lb, -5.0)
    ub_c = np.where(np.isfinite(ub), ub, 5.0)
    N = 2000
    xs = [
        (lb_c + rng.uniform(size=n) * (ub_c - lb_c)).astype(np.float64) for _ in range(N)
    ]

    jac_rows, jac_cols = ev._jac_rows, ev._jac_cols

    # Warm both paths (compile).
    params = ev._current_params()
    _ = ev.evaluate_objective(xs[0])
    _ = ev.evaluate_gradient(xs[0])
    _ = ev.evaluate_constraints(xs[0])
    _ = ev.evaluate_jacobian_values(xs[0])
    jax.block_until_ready(fused_jit(xs[0], params))

    # --- Current per-quantity path: obj+grad+cons+jac per iterate ---
    t0 = time.perf_counter_ns()
    base_vals = []
    for x in xs:
        f = ev.evaluate_objective(x)
        g = ev.evaluate_gradient(x)
        c = ev.evaluate_constraints(x)
        jv = ev.evaluate_jacobian_values(x)
        base_vals.append((f, g, c, jv))
    t_base = (time.perf_counter_ns() - t0) / 1e3

    # --- Fused + device_get once per iterate; slice numpy per quantity ---
    t0 = time.perf_counter_ns()
    fused_vals = []
    for x in xs:
        params = ev._current_params()
        f_j, g_j, c_j, J_j = fused_jit(x, params)
        f, g, c, J = jax.device_get((f_j, g_j, c_j, J_j))
        f = float(f)
        jv = J[jac_rows, jac_cols].astype(np.float64)
        fused_vals.append((f, g, c, jv))
    t_fused = (time.perf_counter_ns() - t0) / 1e3

    # Value identity check (fused must reproduce per-quantity byte-for-byte-ish).
    max_abs = {"f": 0.0, "g": 0.0, "c": 0.0, "J": 0.0}
    for (f0, g0, c0, j0), (f1, g1, c1, j1) in zip(base_vals, fused_vals):
        max_abs["f"] = max(max_abs["f"], abs(f0 - f1))
        max_abs["g"] = max(max_abs["g"], float(np.max(np.abs(g0 - g1))) if g0.size else 0.0)
        max_abs["c"] = max(max_abs["c"], float(np.max(np.abs(c0 - c1))) if c0.size else 0.0)
        max_abs["J"] = max(max_abs["J"], float(np.max(np.abs(j0 - j1))) if j0.size else 0.0)

    return {
        "n_iterates": N,
        "use_sparse_structure": bool(use_sparse),
        "per_quantity_us": t_base,
        "fused_memoized_us": t_fused,
        "per_quantity_us_per_iter": t_base / N,
        "fused_us_per_iter": t_fused / N,
        "speedup_x": t_base / t_fused if t_fused else 0.0,
        "max_abs_diff": max_abs,
    }


def main():
    out = {}
    print("=== PART 1: callback census (one nvs05 solve, 20s) ===", file=sys.stderr)
    out["census"] = _census_solve()
    print(json.dumps(out["census"], indent=2, default=str), file=sys.stderr)

    print("\n=== PART 2: fused prototype probe ===", file=sys.stderr)
    out["fused_probe"] = _fused_prototype_probe()
    print(json.dumps(out["fused_probe"], indent=2, default=str), file=sys.stderr)

    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
