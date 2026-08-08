"""Issue #946 differential panel: flag-free bound-changing change, so the
evidence has to come from a before/after comparison over a corpus.

Runs a deterministic family of convex two-stage MINLPs (binary and integer first
stages, quadratic / conic recourse, equality and inequality coupling, varied
scale) through ``solve_benders`` and records status / objective / bound, in two
arms:

  default   — the recourse NLP keeps Ipopt's ``bound_relax_factor``
  exact     — ``bound_relax_factor = 0`` (the degenerate arm of #946)

The monolithic ``Model.solve()`` optimum is the oracle. Usage::

    python -u differential_panel.py OUT.json

Run it once on the pre-fix tree and once on the post-fix tree, then compare with
``compare_panel.py``. Prints per-instance progress, and exits non-zero if no
instance was actually measured (CLAUDE.md §6/§10).
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np

import discopt.modeling as dm
import discopt.solvers.nlp_pounce as nlp_pounce
from discopt.decomposition.benders import solve_benders


def build(seed: int):
    """A deterministic convex two-stage MINLP. Some seeds are degenerate at a
    first-stage point (the ``<= c * sum(y)`` conic row collapses at y = 0)."""
    rng = np.random.default_rng(seed)
    ny = int(rng.integers(1, 4))
    nx = int(rng.integers(1, 4))
    scale = 10.0 ** rng.integers(-1, 3)
    binary_master = bool(rng.integers(0, 4))  # 3/4 binary, 1/4 general integer
    m = dm.Model(f"panel{seed}")
    if binary_master:
        y = m.binary("y", shape=(ny,))
    else:
        y = m.integer("y", shape=(ny,), lb=0, ub=2)
    x = m.continuous("x", shape=(nx,), lb=0, ub=5)
    m.first_stage(y)
    a = rng.uniform(0.2, 2.0, nx) * scale
    sh = rng.uniform(0, 3, nx)
    cy = rng.uniform(-1, 4, ny) * scale
    linear_obj = bool(rng.integers(0, 3) == 0)
    if linear_obj:
        m.minimize(
            sum(-float(a[j]) * x[j] for j in range(nx)) + sum(float(cy[i]) * y[i] for i in range(ny))
        )
    else:
        m.minimize(
            sum(float(a[j]) * (x[j] - float(sh[j])) ** 2 for j in range(nx))
            + sum(float(cy[i]) * y[i] for i in range(ny))
        )
    sy = sum(y[i] for i in range(ny))
    kind = int(rng.integers(0, 3))
    if kind == 0:
        # Degenerate at sum(y) = 0: the feasible set collapses to x = 0.
        m.subject_to(sum(x[j] * x[j] for j in range(nx)) <= float(rng.uniform(2, 9)) * sy)
    elif kind == 1:
        m.subject_to(sum(x[j] * x[j] for j in range(nx)) <= float(rng.uniform(2, 9)) * sy + 0.1)
    else:
        m.subject_to(sum(x[j] for j in range(nx)) == float(rng.uniform(0, 3)) * sy)
    return m


def _flatten(model, x_dict):
    """Flat vector in ``model._variables`` order from a SolveResult ``x`` dict."""
    if x_dict is None:
        return None
    out = []
    for v in model._variables:
        val = np.asarray(x_dict[v.name], dtype=float).reshape(-1)
        if val.size != v.size:
            return None
        out.extend(val.tolist())
    return np.array(out)


def max_violation(model, x_dict) -> float | None:
    """Largest constraint/box violation of a reported solution.

    The oracle for this panel is the *monolithic* optimum, and a solver that
    returns a point a few 1e-9 outside a degenerate row reports an objective
    below the true optimum (issue #940/#945). Without this number a valid GBD
    bound looks 'unsound' against an infeasible reference.
    """
    from discopt._jax.nlp_evaluator import NLPEvaluator
    from discopt.decomposition.structure import flat_bounds
    from discopt.solvers.nlp_ipopt import _infer_constraint_bounds

    z = _flatten(model, x_dict)
    if z is None:
        return None
    ev = NLPEvaluator(model)
    lb, ub = flat_bounds(model)
    viol = float(max(np.max(lb - z, initial=0.0), np.max(z - ub, initial=0.0), 0.0))
    if ev.n_constraints:
        cl, cu = _infer_constraint_bounds(ev)
        g = np.asarray(ev.evaluate_constraints(z), dtype=float)
        viol = max(viol, float(np.max(g - cu, initial=0.0)), float(np.max(cl - g, initial=0.0)))
    return viol


def run_arm(seed: int, exact_bounds: bool) -> dict:
    real = nlp_pounce.solve_nlp
    calls = {"n": 0, "mu_max": 0.0}

    def patched(problem, x0, options=None):
        opts = dict(options or {})
        if exact_bounds:
            opts["bound_relax_factor"] = 0.0
        res = real(problem, x0, options=opts)
        calls["n"] += 1
        if res.multipliers is not None and len(res.multipliers):
            calls["mu_max"] = max(calls["mu_max"], float(np.max(np.abs(res.multipliers))))
        return res

    nlp_pounce.solve_nlp = patched  # type: ignore[assignment]
    t0 = time.time()
    try:
        r = solve_benders(build(seed), time_limit=60)
    except Exception as exc:  # record, never swallow silently
        nlp_pounce.solve_nlp = real  # type: ignore[assignment]
        return {"seed": seed, "exact": exact_bounds, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        nlp_pounce.solve_nlp = real  # type: ignore[assignment]
    wall = time.time() - t0
    model = build(seed)
    mono = model.solve(time_limit=60)
    return {
        "seed": seed,
        "exact": exact_bounds,
        "status": r.status,
        "objective": r.objective,
        "bound": r.bound,
        "wall": wall,
        "nlp_calls": calls["n"],
        "mu_max": calls["mu_max"],
        "gbd_violation": max_violation(model, r.x),
        "mono_status": mono.status,
        "mono_objective": mono.objective,
        "mono_violation": max_violation(model, mono.x),
    }


def main() -> int:
    out = sys.argv[1]
    seeds = list(range(40))
    rows = []
    for seed in seeds:
        for exact in (False, True):
            row = run_arm(seed, exact)
            rows.append(row)
            print(
                f"seed={seed:3d} exact={int(exact)} "
                f"status={row.get('status', row.get('error'))!s:>16} "
                f"obj={row.get('objective')!s:>22} bound={row.get('bound')!s:>22} "
                f"mono={row.get('mono_objective')!s:>22} mu_max={row.get('mu_max', 0):.3e}",
                flush=True,
            )
    with open(out, "w") as fh:
        json.dump(rows, fh, indent=1)
    print(f"\nmeasured {len(rows)} runs -> {out}")
    if not rows:
        print("PANEL MEASURED NOTHING", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
