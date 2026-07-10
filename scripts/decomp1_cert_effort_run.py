"""DECOMP-1 measurement harness (task #88). Measurement-only: no solver changes.

Runs one instance (a MINLPLib .nl file or the #598 AMP multi4N MILP relaxation)
with instrumentation:
  - counts every McCormick node-LP solve by terminal status (inner impl incl.
    C-43 retries, and outer driver-visible verdicts),
  - counts MILP-path POUNCE bound-recovery attempts/failures (gap decertify),
  - records the (time, global_lower_bound, incumbent_obj) trajectory by
    proxying PyTreeManager.stats(),
  - emits a single RESULT_JSON line on stdout.

Run with DISCOPT_PROFILE=1 in the env to additionally get the Rust simplex
pivot/phase dumps on stderr (parsed by the parent).
"""

import argparse
import json
import logging
import sys
import time
from collections import Counter


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nl", default=None)
    ap.add_argument("--amp-milp", action="store_true")
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="LOG %(relativeCreated)d %(name)s %(message)s",
    )

    import discopt._jax.mccormick_lp as mlp
    import discopt.solver as ds
    import numpy as np  # noqa: F401

    lp_counts: Counter = Counter()  # inner _solve_at_node_impl statuses
    outer_counts: Counter = Counter()  # driver-visible solve_at_node verdicts

    def _key(res):
        k = res.status
        if k == "optimal" and getattr(res, "lower_bound", None) is None:
            k = "optimal_nobound"  # declined bound -> driver must branch
        return k

    _orig_impl = mlp.MccormickLPRelaxer._solve_at_node_impl

    def _impl(self, *a, **k):
        res = _orig_impl(self, *a, **k)
        lp_counts[_key(res)] += 1
        return res

    mlp.MccormickLPRelaxer._solve_at_node_impl = _impl

    _orig_outer = mlp.MccormickLPRelaxer.solve_at_node

    def _outer(self, *a, **k):
        res = _orig_outer(self, *a, **k)
        outer_counts[_key(res)] += 1
        return res

    mlp.MccormickLPRelaxer.solve_at_node = _outer

    # Non-LP node-bound engines: count calls so each instance's dominant
    # per-node bounder is identified (pure-integer models route to
    # alphaBB/interval, not the McCormick LP).
    engine_counts: Counter = Counter()

    _orig_ab = ds._compute_alphabb_bound

    def _ab(*a, **k):
        engine_counts["alphabb"] += 1
        return _orig_ab(*a, **k)

    ds._compute_alphabb_bound = _ab

    _orig_iv = ds._compute_interval_bound

    def _iv(*a, **k):
        engine_counts["interval"] += 1
        return _orig_iv(*a, **k)

    ds._compute_interval_bound = _iv

    try:
        import discopt._jax.mccormick_nlp as mnlp

        _orig_mcb = mnlp.solve_mccormick_batch

        def _mcb(*a, **k):
            r = _orig_mcb(*a, **k)
            engine_counts["mccormick_nlp_batch_calls"] += 1
            try:
                engine_counts["mccormick_nlp_batch_nodes"] += int(len(r))
            except Exception:
                pass
            return r

        mnlp.solve_mccormick_batch = _mcb
    except Exception:
        pass

    # MILP-path node-bound recovery (iteration-limit/numerical node LP exits):
    # each attempt = one node whose LP bound was untrusted; a None return
    # decertifies the gap (#598 failure class).
    recover = {"attempts": 0, "failed": 0}
    if hasattr(ds, "_pounce_recover_node_bound"):
        _orig_rec = ds._pounce_recover_node_bound

        def _rec(*a, **k):
            recover["attempts"] += 1
            r = _orig_rec(*a, **k)
            if r is None:
                recover["failed"] += 1
            return r

        ds._pounce_recover_node_bound = _rec

    # Bound/incumbent trajectory via a PyTreeManager proxy. stats() is called
    # once per driver batch iteration, so overhead is negligible vs node LPs.
    traj: list = []
    t0 = time.perf_counter()
    _RealTree = ds.PyTreeManager

    class TreeProxy:
        def __init__(self, *a, **k):
            object.__setattr__(self, "_t", _RealTree(*a, **k))

        def __getattr__(self, name):
            real = object.__getattribute__(self, "_t")
            attr = getattr(real, name)
            if name == "stats":

                def stats(*a, **k):
                    s = attr(*a, **k)
                    try:
                        inc = real.incumbent()
                        inc_obj = None if inc is None else float(inc[1])
                    except Exception:
                        inc_obj = None
                    glb = None
                    try:
                        glb = s.get("global_lower_bound")
                        glb = None if glb is None else float(glb)
                    except Exception:
                        pass
                    traj.append((time.perf_counter() - t0, glb, inc_obj))
                    return s

                return stats
            return attr

    ds.PyTreeManager = TreeProxy

    scipy_contrast = None
    if args.amp_milp:
        from discopt import Model
        from discopt._jax.discretization import initialize_partitions
        from discopt._jax.milp_relaxation import build_milp_relaxation
        from discopt._jax.model_utils import flat_variable_bounds
        from discopt._jax.term_classifier import classify_nonlinear_terms

        # Alpine examples/MINLPs/multi.jl:multi4N, n=2, exprmode=1 (issue #598).
        m0 = Model("alpine_multi4N_2_1")
        size = 7
        x = m0.continuous("x", lb=0.1, ub=4.0, shape=(size,))
        obj = None
        for i in range(0, size - 1, 3):
            term = x[i] * x[i + 1] * x[i + 2] * x[i + 3]
            obj = term if obj is None else obj + term
            m0.subject_to(x[i] + x[i + 1] + x[i + 2] + x[i + 3] <= 4.0)
        m0.maximize(obj)

        terms = classify_nonlinear_terms(m0)
        flat_lb, flat_ub = flat_variable_bounds(m0)
        state = initialize_partitions(
            terms.partition_candidates,
            lb=[flat_lb[i] for i in terms.partition_candidates],
            ub=[flat_ub[i] for i in terms.partition_candidates],
            n_init=2,
        )
        model, _varmap = build_milp_relaxation(m0, terms, state, incumbent=None)

        # Identical matrix through scipy's HiGHS MILP for the node contrast.
        # ``model`` is a MilpRelaxationModel wrapper: min c'x s.t. A_ub x <= b_ub,
        # bounds, integrality (already in the internal min sense).
        try:
            import scipy.sparse as sp
            from scipy.optimize import Bounds, LinearConstraint
            from scipy.optimize import milp as scipy_milp

            c = np.asarray(model._c, dtype=float)
            A = sp.csr_matrix(model._A_ub) if model._A_ub is not None else None
            b = np.asarray(model._b_ub, dtype=float) if model._b_ub is not None else None
            xl = np.array([lo for lo, _ in model._bounds], dtype=float)
            xu = np.array([hi for _, hi in model._bounds], dtype=float)
            integrality = (
                np.asarray(model._integrality, dtype=float)
                if model._integrality is not None
                else np.zeros(c.shape[0])
            )
            cons = [LinearConstraint(A, -np.inf, b)] if A is not None and A.shape[0] else []
            t_h = time.perf_counter()
            hres = scipy_milp(c=c, constraints=cons, bounds=Bounds(xl, xu), integrality=integrality)
            scipy_contrast = {
                "status": int(hres.status),
                "message": str(hres.message),
                "objective_min_sense": (
                    None if hres.fun is None else float(hres.fun + model._obj_offset)
                ),
                "mip_node_count": int(getattr(hres, "mip_node_count", -1)),
                "mip_dual_bound": (
                    None
                    if getattr(hres, "mip_dual_bound", None) is None
                    else float(hres.mip_dual_bound)
                ),
                "wall": time.perf_counter() - t_h,
                "n_cols": int(c.shape[0]),
                "n_rows": 0 if A is None else int(A.shape[0]),
                "n_int": int((integrality > 0).sum()),
            }
        except Exception as exc:  # measurement-only: record, don't die
            scipy_contrast = {"error": repr(exc)}
    else:
        from discopt.modeling.core import from_nl

        model = from_nl(args.nl)

    t_solve = time.perf_counter()
    result = model.solve(time_limit=args.time_limit)
    wall = time.perf_counter() - t_solve

    def _get(attr, default=None):
        return getattr(result, attr, default)

    # Compress trajectory: keep points where glb or incumbent changed.
    slim = []
    last = (object(), object())
    for t, glb, inc in traj:
        if (glb, inc) != last:
            slim.append([round(t, 3), glb, inc])
            last = (glb, inc)
    if traj:
        t, glb, inc = traj[-1]
        if slim and slim[-1][0] != round(t, 3):
            slim.append([round(t, 3), glb, inc])

    out = {
        "name": args.name or (args.nl or "amp_multi4n_milp"),
        "status": result.status,
        "objective": result.objective,
        "bound": result.bound,
        "gap": _get("gap"),
        "gap_certified": bool(_get("gap_certified", False)),
        "node_count": int(_get("node_count", -1) or -1),
        "wall_time": float(_get("wall_time", wall) or wall),
        "harness_wall": wall,
        "root_bound": _get("root_bound"),
        "root_gap": _get("root_gap"),
        "root_time": _get("root_time"),
        "safe_bound": _get("safe_bound"),
        "farkas_certified": bool(_get("farkas_certified", False)),
        "convex_fast_path": bool(_get("convex_fast_path", False)),
        "nlp_bb": bool(_get("nlp_bb", False)),
        "solver_stats": _get("solver_stats"),
        "mccormick_lp_impl_counts": dict(lp_counts),
        "mccormick_lp_outer_counts": dict(outer_counts),
        "milp_recover": recover,
        "engine_counts": dict(engine_counts),
        "traj": slim,
        "n_traj_snapshots": len(traj),
        "scipy_highs": scipy_contrast,
    }
    print("RESULT_JSON: " + json.dumps(out))


if __name__ == "__main__":
    main()
