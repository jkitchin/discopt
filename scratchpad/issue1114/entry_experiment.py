"""#1114 entry experiment (CLAUDE.md §4): does alphaBB ever WIN a node the
reduced-space engine also bounded?

`_use_alphabb` keys on `_mc_lp_relaxer is None`, which is exactly the hidden-function
(`dm.custom` / CustomCall) class where the reduced-space engine is forced on. Both
engines therefore run at every node and are combined with `max()`. This probe counts,
per model and per node box:

  * nodes where alphaBB returned a finite bound,
  * nodes where the reduced engine returned a bound,
  * nodes where BOTH did, and of those, nodes where `alphabb_lb > reduced_lb`
    (strictly, beyond a relative tolerance).

KILL CRITERION (from the issue): if `alphabb_lb > reduced_lb` on a non-trivial
fraction of nodes, the redundancy claim is false and #1114 must be re-scoped to the
root alpha estimate alone.

Usage: python -u entry_experiment.py [max_nodes] [time_limit]

The corpus is a FAMILY of MCBox-traceable CustomCall models with different structures
and dimensions (§2: the class, not an instance). Exceptions are never swallowed (§7);
the executed-comparison count is printed and a zero count exits non-zero (§6).
"""

from __future__ import annotations

import json
import sys

import numpy as np

import discopt
import discopt.modeling as dm
import discopt.solver as solver_mod
from discopt._relax import mccormick_subgradient as _mcs
from discopt._relax.mcbox import MCBox

import jax.numpy as jnp

print(f"discopt.__file__={discopt.__file__}", flush=True)

MAX_NODES = int(sys.argv[1]) if len(sys.argv) > 1 else 200
TIME_LIMIT = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0


# --- MCBox/numpy dual-dispatch helpers (the canonical notebook idiom) --------- #
def _exp(x):
    return x.exp() if isinstance(x, MCBox) else jnp.exp(x)


def _log(x):
    return x.log() if isinstance(x, MCBox) else jnp.log(x)


def _sqrt(x):
    return x.sqrt() if isinstance(x, MCBox) else jnp.sqrt(x)


def _tanh(x):
    return x.tanh() if isinstance(x, MCBox) else jnp.tanh(x)


def _sigmoid(x):
    return x.sigmoid() if isinstance(x, MCBox) else 1.0 / (1.0 + jnp.exp(-x))


def _atan(x):
    return x.atan() if isinstance(x, MCBox) else jnp.arctan(x)


def _sinh(x):
    return x.sinh() if isinstance(x, MCBox) else jnp.sinh(x)


def _model_exp_product():
    m = dm.Model("exp_product")
    x = m.continuous("x", 2, lb=[0.1, 0.1], ub=[2.0, 2.0])
    f = dm.custom(lambda a, b: a * _exp(-b) + b * _exp(-a), name="f")
    m.minimize(f(x[0], x[1]))
    m.subject_to(x[0] + x[1] >= 1.0)
    return m


def _model_bilinear_chain():
    m = dm.Model("bilinear_chain")
    x = m.continuous("x", 3, lb=[-1.0, -1.0, -1.0], ub=[1.5, 1.5, 1.5])
    f = dm.custom(lambda a, b, c: a * b + b * c - 0.7 * a * c, name="f")
    m.minimize(f(x[0], x[1], x[2]))
    m.subject_to(x[0] + x[1] + x[2] <= 2.0)
    return m


def _model_log_ratio():
    m = dm.Model("log_ratio")
    x = m.continuous("x", 2, lb=[0.5, 0.5], ub=[3.0, 3.0])
    f = dm.custom(lambda a, b: _log(a + 0.5 * b) - a / (b + 0.25), name="f")
    m.minimize(f(x[0], x[1]))
    m.subject_to(x[0] - x[1] <= 1.0)
    return m


def _model_sqrt_sum():
    m = dm.Model("sqrt_sum")
    x = m.continuous("x", 3, lb=[0.05, 0.05, 0.05], ub=[4.0, 4.0, 4.0])
    f = dm.custom(lambda a, b, c: _sqrt(a) * b - 0.4 * _sqrt(b * c + 0.1), name="f")
    m.minimize(f(x[0], x[1], x[2]))
    m.subject_to(x[0] + x[2] >= 1.0)
    return m


def _model_tanh_net():
    m = dm.Model("tanh_net")
    x = m.continuous("x", 2, lb=[-1.5, -1.5], ub=[1.5, 1.5])
    f = dm.custom(
        lambda a, b: 1.3 * _tanh(0.7 * a - 0.4 * b + 0.1) - 0.8 * _tanh(-0.3 * a + 0.9 * b),
        name="f",
    )
    m.minimize(f(x[0], x[1]))
    return m


def _model_sigmoid_mix():
    m = dm.Model("sigmoid_mix")
    x = m.continuous("x", 3, lb=[-2.0, -2.0, -2.0], ub=[2.0, 2.0, 2.0])
    f = dm.custom(
        lambda a, b, c: _sigmoid(a * b) + 0.5 * _sigmoid(b - c) - 0.3 * a * c, name="f"
    )
    m.minimize(f(x[0], x[1], x[2]))
    m.subject_to(x[0] + x[1] + x[2] >= -1.0)
    return m


def _model_atan_sinh():
    m = dm.Model("atan_sinh")
    x = m.continuous("x", 2, lb=[-1.0, -1.0], ub=[1.0, 1.0])
    f = dm.custom(lambda a, b: _atan(3.0 * a * b) + 0.2 * _sinh(a - b), name="f")
    m.minimize(f(x[0], x[1]))
    return m


def _model_pow_mix():
    m = dm.Model("pow_mix")
    x = m.continuous("x", 3, lb=[0.2, 0.2, 0.2], ub=[2.5, 2.5, 2.5])
    f = dm.custom(lambda a, b, c: a**3 - 2.0 * a * b + b**2 * c - 0.5 * c**2, name="f")
    m.minimize(f(x[0], x[1], x[2]))
    m.subject_to(x[0] + x[1] >= 0.8)
    return m


MODELS = [
    _model_exp_product,
    _model_bilinear_chain,
    _model_log_ratio,
    _model_sqrt_sum,
    _model_tanh_net,
    _model_sigmoid_mix,
    _model_atan_sinh,
    _model_pow_mix,
]

_orig_alphabb = solver_mod._compute_alphabb_bound
_orig_reduced = _mcs.reduced_mccormick_lp_bound

REL_TOL = 1e-9


def _key(lb, ub, k):
    a = np.asarray(lb, dtype=np.float64)[:k]
    b = np.asarray(ub, dtype=np.float64)[:k]
    return (a.tobytes(), b.tobytes())


rows = []
comparisons = 0
for factory in MODELS:
    model = factory()
    alpha_calls: dict = {}
    reduced_calls: dict = {}

    def _alphabb_probe(evaluator, mdl, expr, node_lb, node_ub, _o=_orig_alphabb, _a=alpha_calls):
        val = _o(evaluator, mdl, expr, node_lb, node_ub)
        _a.setdefault(_key(node_lb, node_ub, len(np.asarray(node_lb))), []).append(float(val))
        return val

    def _reduced_probe(mdl, lo, hi, *args, _o=_orig_reduced, _r=reduced_calls, **kwargs):
        res = _o(mdl, lo, hi, *args, **kwargs)
        _r.setdefault(_key(lo, hi, len(np.asarray(lo))), []).append(
            (res.status, None if res.bound is None else float(res.bound))
        )
        return res

    solver_mod._compute_alphabb_bound = _alphabb_probe
    _mcs.reduced_mccormick_lp_bound = _reduced_probe
    try:
        result = model.solve(max_nodes=MAX_NODES, time_limit=TIME_LIMIT)
    finally:
        solver_mod._compute_alphabb_bound = _orig_alphabb
        _mcs.reduced_mccormick_lp_bound = _orig_reduced

    # Pair by node box: the reduced engine is called on the ORIGINAL-variable slice,
    # alphaBB on the (possibly lifted) node box, so match on the shorter prefix.
    n_alpha_finite = 0
    n_reduced_bound = 0
    n_both = 0
    n_alpha_wins = 0
    worst_win = 0.0
    for rkey, rvals in reduced_calls.items():
        lb_r = np.frombuffer(rkey[0], dtype=np.float64)
        k = len(lb_r)
        rbounds = [v for (s, v) in rvals if s == "optimal" and v is not None and np.isfinite(v)]
        if rbounds:
            n_reduced_bound += 1
        akey = None
        for cand in alpha_calls:
            if np.frombuffer(cand[0], dtype=np.float64)[:k].tobytes() == rkey[0] and (
                np.frombuffer(cand[1], dtype=np.float64)[:k].tobytes() == rkey[1]
            ):
                akey = cand
                break
        if akey is None:
            continue
        avals = [v for v in alpha_calls[akey] if np.isfinite(v)]
        if avals:
            n_alpha_finite += 1
        if avals and rbounds:
            n_both += 1
            a_best, r_best = max(avals), max(rbounds)
            comparisons += 1
            if a_best > r_best + REL_TOL * (1.0 + abs(r_best)):
                n_alpha_wins += 1
                worst_win = max(worst_win, a_best - r_best)

    row = {
        "model": model.name if hasattr(model, "name") else str(factory.__name__),
        "status": result.status,
        "nodes": int(result.node_count or 0),
        "bound": None if result.bound is None else float(result.bound),
        "objective": None if result.objective is None else float(result.objective),
        "alpha_boxes": len(alpha_calls),
        "reduced_boxes": len(reduced_calls),
        "boxes_alpha_finite": n_alpha_finite,
        "boxes_reduced_bound": n_reduced_bound,
        "boxes_both": n_both,
        "boxes_alpha_wins": n_alpha_wins,
        "worst_alpha_margin": worst_win,
    }
    rows.append(row)
    print(json.dumps(row), flush=True)

tot_both = sum(r["boxes_both"] for r in rows)
tot_wins = sum(r["boxes_alpha_wins"] for r in rows)
print(
    f"--- TOTAL both={tot_both} alpha_wins={tot_wins} "
    f"frac={0.0 if tot_both == 0 else tot_wins / tot_both:.4f}",
    flush=True,
)
print(f"comparisons={comparisons}", flush=True)
if comparisons == 0:
    print("PROBE FIRED NOTHING — no node box had both bounds; the experiment is void.")
    sys.exit(2)
