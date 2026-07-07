"""cert:T2.2 A/B differential harness — persistent+warm OBBT vs cold-seam OBBT.

Verifies the bound-neutrality of the persistent per-sweep OBBT LP + warm-started
probes (T2.2 (a)+(b), in ``discopt/_jax/obbt.py``): within a sweep the std-form
CSC is assembled once and each probe warm-starts from the previous probe's
optimal basis. Because probes differ only in the objective over a (weakly)
shrinking box, the warm basis is usually still primal-feasible and the Rust
``solve_lp_cols_warm`` path finishes in a primal phase-2 — else it falls to the
trusted cold two-phase solve. The optimum is the exact vertex either way, so the
*applied tightenings* must be bit-identical to the pre-T2.2 cold path.

The A/B compares the default (persistent+warm) path against the seam-based cold
path (``_simplex_available`` forced False), which reproduces the old behavior. On
>=200 sampled probes across >=5 instances it asserts warm bound == cold bound to
1e-9 and identical tightenings. In-repo toy models always run; the larger
MINLPLib panel (ex1252a/gear4/st_e38/ex1252/gear) runs when the snapshot is
present (``~/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl``).

Standalone runnable:
    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python discopt_benchmarks/scripts/t22_obbt_ab.py
"""

from __future__ import annotations

import os
import os.path as osp

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt._jax.obbt as obbt_mod
import discopt.modeling as dm
import numpy as np
from discopt._jax.obbt import run_obbt_on_relaxation
from discopt.modeling.core import Model

_SNAP = osp.expanduser("~/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl")


def _build_relaxation(model: Model):
    from discopt._jax.discretization import initialize_partitions
    from discopt._jax.milp_relaxation import build_milp_relaxation
    from discopt._jax.term_classifier import classify_nonlinear_terms

    terms = classify_nonlinear_terms(model)
    state = initialize_partitions([], lb=[], ub=[], n_init=2)
    milp, _varmap = build_milp_relaxation(model, terms, state, incumbent=None)
    return milp


def _bilinear() -> Model:
    m = Model()
    x = m.continuous("x", lb=-2.0, ub=3.0)
    y = m.continuous("y", lb=-1.0, ub=4.0)
    z = m.continuous("z", lb=-10.0, ub=10.0)
    m.subject_to(x * y <= 5.0)
    m.subject_to(x + y + z <= 6.0)
    m.subject_to(z >= x * y - 2.0)
    m.minimize(z)
    return m


def _cubic() -> Model:
    m = Model()
    x = m.continuous("x", lb=0.5, ub=4.0)
    y = m.continuous("y", lb=0.5, ub=4.0)
    m.subject_to(x * y * x <= 20.0)
    m.subject_to(x + y <= 6.0)
    m.minimize(x + y)
    return m


def _ratio() -> Model:
    m = Model()
    x = m.continuous("x", lb=1.0, ub=10.0)
    y = m.continuous("y", lb=1.0, ub=10.0)
    m.subject_to(x / y <= 5.0)
    m.subject_to(x + y <= 15.0)
    m.minimize(x - y)
    return m


def _mixed() -> Model:
    m = Model()
    x = m.continuous("x", lb=-3.0, ub=3.0)
    y = m.integer("y", lb=-3, ub=3)
    z = m.continuous("z", lb=-20.0, ub=20.0)
    m.subject_to(x * y <= 4.0)
    m.subject_to(z >= x * x - y)
    m.subject_to(x + y + z <= 10.0)
    m.minimize(z)
    return m


def _wide() -> Model:
    m = Model()
    xs = [m.continuous(f"x{i}", lb=-2.0, ub=2.0) for i in range(4)]
    m.subject_to(xs[0] * xs[1] + xs[2] * xs[3] <= 3.0)
    m.subject_to(sum(xs) <= 4.0)
    m.subject_to(xs[0] * xs[2] >= -3.0)
    m.minimize(xs[0] + xs[3])
    return m


def _nl_loader(stem):
    def load():
        return dm.from_nl(osp.join(_SNAP, f"{stem}.nl"))

    return load


def _panel():
    panel = {
        "bilinear": _bilinear,
        "cubic": _cubic,
        "ratio": _ratio,
        "mixed": _mixed,
        "wide": _wide,
    }
    if osp.isdir(_SNAP):
        for stem in ("ex1252a", "gear4", "st_e38", "ex1252", "gear"):
            if osp.isfile(osp.join(_SNAP, f"{stem}.nl")):
                panel[stem] = _nl_loader(stem)
    return panel


def _maxdiff(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    both_inf = ~np.isfinite(a) & ~np.isfinite(b) & (np.sign(a) == np.sign(b))
    d = np.abs(a - b)
    d[both_inf] = 0.0
    return float(np.max(d)) if d.size else 0.0


def _run(model, n_orig, cutoff, warm):
    orig = obbt_mod._simplex_available
    if not warm:
        obbt_mod._simplex_available = lambda: False
    try:
        rel = _build_relaxation(model)
        return run_obbt_on_relaxation(
            rel, n_orig=n_orig, time_limit_per_lp=5.0, incumbent_cutoff=cutoff
        )
    finally:
        obbt_mod._simplex_available = orig


def main() -> int:
    assert obbt_mod._simplex_available(), "Rust simplex binding required"
    panel = _panel()
    total_probes = 0
    max_lb, max_ub = 0.0, 0.0
    failures = []
    n_configs = 0
    for name, fn in panel.items():
        model = fn()
        n_orig = sum(v.size for v in model._variables)
        for cutoff in (None, 100.0, 5.0):
            warm = _run(model, n_orig, cutoff, warm=True)
            cold = _run(model, n_orig, cutoff, warm=False)
            lb_diff = _maxdiff(warm.tightened_lb, cold.tightened_lb)
            ub_diff = _maxdiff(warm.tightened_ub, cold.tightened_ub)
            max_lb, max_ub = max(max_lb, lb_diff), max(max_ub, ub_diff)
            total_probes += warm.n_lp_solves
            n_configs += 1
            ok = (
                lb_diff <= 1e-9
                and ub_diff <= 1e-9
                and warm.n_tightened == cold.n_tightened
                and warm.n_lp_solves == cold.n_lp_solves
            )
            if not ok:
                failures.append((name, cutoff, lb_diff, ub_diff))
            print(
                f"[{'OK' if ok else 'MISMATCH'}] {name:9s} cutoff={str(cutoff):5s} "
                f"probes={warm.n_lp_solves:3d} lb_diff={lb_diff:.2e} ub_diff={ub_diff:.2e} "
                f"n_tight(w/c)={warm.n_tightened}/{cold.n_tightened}"
            )
    print("\n=== SUMMARY ===")
    print(f"models: {len(panel)}  configs: {n_configs}  warm probe LP solves: {total_probes}")
    print(f"max |lb_diff|={max_lb:.3e}  max |ub_diff|={max_ub:.3e}")
    if failures:
        print(f"NOT NEUTRAL — {len(failures)} mismatches: {failures}")
        return 1
    print("DIFFERENTIAL NEUTRAL (warm == cold to 1e-9; tightenings identical)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
