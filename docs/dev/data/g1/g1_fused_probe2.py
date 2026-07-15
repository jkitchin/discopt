"""G1 corrected fused prototype: fuse along MEASURED co-occurrence.

Census (part 1) showed the dominant iterate pattern is objective+constraints
only (87.8% of distinct iterates = line-search trial points). Fusing all four
(f,g,c,J) computes the expensive gradient+Jacobian at those trial points -> net
loss. Correct design: fuse (f,c) and (g,J) separately, memoized per iterate.

This probe replays a REALISTIC access sequence (proportions from the census)
through: (a) current per-quantity path, (b) memoized dual-fused path. Reports
callback-overhead wall + value identity.
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np


NL_PATH = "python/tests/data/minlplib_nl/nvs05.nl"


def main():
    import jax
    import jax.numpy as jnp

    import discopt._jax.nlp_evaluator as nev
    from discopt.modeling.core import from_nl

    model = from_nl(NL_PATH)
    ev = nev.cached_evaluator(model)
    n = ev.n_variables
    ev._ensure_coo_cache()
    jac_rows, jac_cols = ev._jac_rows, ev._jac_cols

    obj_fn, grad_fn = ev._obj_fn_jit, ev._grad_fn_jit
    cons_fn, jac_fn = ev._cons_fn_jit, ev._jac_fn_jit

    def fused_fc(x, params):
        return obj_fn(x, params), cons_fn(x, params)

    def fused_gj(x, params):
        return grad_fn(x, params), jac_fn(x, params)

    fused_fc_jit = jax.jit(fused_fc)
    fused_gj_jit = jax.jit(fused_gj)

    rng = np.random.default_rng(11)
    lb, ub = ev.variable_bounds
    lb_c = np.where(np.isfinite(lb), lb, -5.0)
    ub_c = np.where(np.isfinite(ub), ub, 5.0)

    # Realistic sequence: ~88% (obj,cons)-only, ~12% full (obj,grad,cons,jac).
    N = 5000
    seq = []
    for _ in range(N):
        x = (lb_c + rng.uniform(size=n) * (ub_c - lb_c)).astype(np.float64)
        full = rng.random() < 0.12
        seq.append((x, full))

    # Warm compile.
    p = ev._current_params()
    _ = ev.evaluate_objective(seq[0][0])
    _ = ev.evaluate_constraints(seq[0][0])
    _ = ev.evaluate_gradient(seq[0][0])
    _ = ev.evaluate_jacobian_values(seq[0][0])
    jax.block_until_ready(fused_fc_jit(seq[0][0], p))
    jax.block_until_ready(fused_gj_jit(seq[0][0], p))

    # --- (a) current per-quantity path ---
    t0 = time.perf_counter_ns()
    base = []
    for x, full in seq:
        f = ev.evaluate_objective(x)
        c = ev.evaluate_constraints(x)
        if full:
            g = ev.evaluate_gradient(x)
            jv = ev.evaluate_jacobian_values(x)
        else:
            g = jv = None
        base.append((f, c, g, jv))
    t_base = (time.perf_counter_ns() - t0) / 1e3

    # --- (b) memoized dual-fused path (params cached per iterate too) ---
    t0 = time.perf_counter_ns()
    fused = []
    for x, full in seq:
        params = ev._current_params()
        f_j, c_j = fused_fc_jit(x, params)
        f, c = jax.device_get((f_j, c_j))
        f = float(f)
        if full:
            g_j, J_j = fused_gj_jit(x, params)
            g, J = jax.device_get((g_j, J_j))
            jv = J[jac_rows, jac_cols].astype(np.float64)
        else:
            g = jv = None
        fused.append((f, c, g, jv))
    t_fused = (time.perf_counter_ns() - t0) / 1e3

    # value identity
    md = {"f": 0.0, "c": 0.0, "g": 0.0, "J": 0.0}
    for (f0, c0, g0, j0), (f1, c1, g1, j1) in zip(base, fused):
        md["f"] = max(md["f"], abs(f0 - f1))
        md["c"] = max(md["c"], float(np.max(np.abs(c0 - c1))) if c0.size else 0.0)
        if g0 is not None:
            md["g"] = max(md["g"], float(np.max(np.abs(g0 - g1))))
            md["J"] = max(md["J"], float(np.max(np.abs(j0 - j1))))

    # Also measure the pure (f,c)-only overhead reduction (the 88% majority).
    fc_seq = [x for x, _ in seq]
    t0 = time.perf_counter_ns()
    for x in fc_seq:
        ev.evaluate_objective(x)
        ev.evaluate_constraints(x)
    t_fc_base = (time.perf_counter_ns() - t0) / 1e3
    t0 = time.perf_counter_ns()
    for x in fc_seq:
        params = ev._current_params()
        f_j, c_j = fused_fc_jit(x, params)
        f, c = jax.device_get((f_j, c_j))
        float(f)
    t_fc_fused = (time.perf_counter_ns() - t0) / 1e3

    out = {
        "n_iterates": N,
        "mixed_88_12": {
            "per_quantity_us": t_base,
            "fused_us": t_fused,
            "speedup_x": t_base / t_fused if t_fused else 0.0,
        },
        "fc_only_majority": {
            "per_quantity_us": t_fc_base,
            "fused_us": t_fc_fused,
            "speedup_x": t_fc_base / t_fc_fused if t_fc_fused else 0.0,
        },
        "max_abs_diff": md,
    }
    print(json.dumps(out, indent=2), file=sys.stderr)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
