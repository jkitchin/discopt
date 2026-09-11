#!/usr/bin/env python
"""Tape vs discopt-doe's shipped compute_fim, on the SAME experiments (#1215).

Closes the #727 loop: §33's tape result was measured on a synthetic collocation
replica. This runs the tape against ``discopt.doe.fim.compute_fim`` itself, on
the shipped Langmuir and kinetics Experiment shapes, and checks the tape's
dY/dtheta against ``FIMResult.jacobian`` -- the real path's own answer.

Kill criterion: max relative error > 1e-9 against the shipped Jacobian.

Requires the ``discopt-doe`` plugin (``pip install discopt-doe``); skips with a
clear message when it is absent, so this is never a silent no-op.

Measured 2026-09-09 on `main` @ c052e85, load 0.29, validated against
``FIMResult.jacobian`` over 1500 Jacobian entries:

================  ============  ===========  ==========  =========  =============
case              doe median    tape build   tape eval   speedup    max rel err
================  ============  ===========  ==========  =========  =============
langmuir n=50          354.8 ms      3.54 ms      9.5 us       100x     8.882e-16
langmuir n=200        1462.8 ms     12.51 ms     22.6 us       117x     8.882e-16
kinetics n=50          482.5 ms      5.45 ms     13.8 us        88x     2.528e-16
kinetics n=200        1888.8 ms     20.12 ms     38.6 us        94x     3.767e-16
================  ============  ===========  ==========  =========  =============

Agreement is float roundoff (~1e-16), not merely inside the 1e-9 bar. The
speedup column includes the tape build, so it is the honest end-to-end figure
for one FIM evaluation from a fresh model.

NOTE ON WHAT THIS PROVES: agreement is against what discopt-doe already
computes, so it shows the tape reproduces the shipped answer -- not that either
is correct in absolute terms. ``compute_fim(method="finite_difference")`` is the
independent third leg and is NOT exercised here.

§6: prints the number of Jacobian entries compared per case; exits non-zero if 0
or on disagreement.
§8: prints the module file it loaded.
§9: load gate reported; both arms timed over 20 calls, medians compared.

Usage::

    python -u discopt_benchmarks/scripts/issue1215_real_fim_validation.py
"""

import os
import statistics
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm
import discopt.modeling.core as core
import numpy as np

try:
    from discopt.doe.fim import compute_fim
except ImportError:  # pragma: no cover - plugin absent
    print("SKIP: discopt-doe is not installed (pip install discopt-doe)")
    raise SystemExit(0) from None
from discopt.estimate import Experiment, ExperimentModel

print(f"# discopt loaded from: {core.__file__}")
print(f"# load gate: {os.getloadavg()[0]:.2f}")


class LangmuirExperiment(Experiment):
    def __init__(self, n, sigma=0.05):
        self.c_data = list(np.geomspace(0.1, 10.0, n))
        self.sigma = float(sigma)

    def create_model(self, **kwargs):
        m = dm.Model("langmuir")
        qm = m.continuous("qm", lb=0.01, ub=20.0)
        k_aff = m.continuous("K", lb=1e-4, ub=1e4)  # noqa: N806 -- Langmuir K
        r, e = {}, {}
        for i, c in enumerate(self.c_data):
            r[f"q_{i}"] = qm * k_aff * c / (1.0 + k_aff * c)
            e[f"q_{i}"] = self.sigma
        return ExperimentModel(m, {"qm": qm, "K": k_aff}, {}, r, e)


class KineticsExperiment(Experiment):
    def __init__(self, n, sigma=0.05):
        self.t = list(np.linspace(0.1, 5.0, n))
        self.sigma = float(sigma)

    def create_model(self, **kwargs):
        m = dm.Model("kinetics")
        a = m.continuous("A", lb=0.1, ub=20.0)
        k = m.continuous("k", lb=0.01, ub=5.0)
        ea = m.continuous("Ea", lb=0.1, ub=10.0)
        ci = m.continuous("Cinf", lb=0.0, ub=5.0)
        r, e = {}, {}
        for i, ti in enumerate(self.t):
            r[f"y_{i}"] = a * dm.exp(-k * dm.exp(-ea / 2.0) * ti) + ci
            e[f"y_{i}"] = self.sigma
        return ExperimentModel(m, {"A": a, "k": k, "Ea": ea, "Cinf": ci}, {}, r, e)


CASES = [
    ("langmuir n=50", LangmuirExperiment(50), {"qm": 5.0, "K": 1.0}),
    ("langmuir n=200", LangmuirExperiment(200), {"qm": 5.0, "K": 1.0}),
    ("kinetics n=50", KineticsExperiment(50), {"A": 5.0, "k": 0.5, "Ea": 1.0, "Cinf": 0.5}),
    ("kinetics n=200", KineticsExperiment(200), {"A": 5.0, "k": 0.5, "Ea": 1.0, "Cinf": 0.5}),
]

import pounce  # noqa: E402
from discopt._nl_expr_compiler import compile_to_nl_expr  # noqa: E402
from discopt.parametric import variable_slices  # noqa: E402

N_REP = 20
total_cmp = 0
print(
    f"\n{'case':16s} {'doe median':>11s} {'tape build':>11s} {'tape eval':>11s} "
    f"{'speedup':>9s} {'max rel err':>12s}"
)
for label, exp, theta in CASES:
    # ── shipped path ──
    res = compute_fim(exp, theta)  # warm
    J_doe = np.asarray(res.jacobian)
    doe_ts = []
    for _ in range(N_REP):
        t0 = time.perf_counter()
        compute_fim(exp, theta)
        doe_ts.append(time.perf_counter() - t0)
    doe_med = statistics.median(doe_ts)

    # ── tape path, same model ──
    em = exp.create_model(**theta)
    m = em.model
    slices = variable_slices(m)
    n_x = max(sl.stop for sl in slices.values())
    lo = np.concatenate([np.asarray(v.lb).reshape(-1) for v in m._variables])
    hi = np.concatenate([np.asarray(v.ub).reshape(-1) for v in m._variables])
    x0 = np.zeros(n_x)
    for name, val in theta.items():
        x0[slices[em.unknown_parameters[name].name].start] = val
    theta_cols = [slices[em.unknown_parameters[n].name].start for n in theta]

    t0 = time.perf_counter()
    exprs = [compile_to_nl_expr(em.responses[k], m) for k in em.responses]
    prob = pounce.build_nl_problem(
        n_x, pounce.NlExpr.const_(0.0), constraints=exprs, x_l=lo, x_u=hi
    )
    t_build = time.perf_counter() - t0
    jr, jc = (np.asarray(a) for a in prob.jacobian_structure())

    n_rows = len(exprs)

    def tape_jac(x, prob=prob, jr=jr, jc=jc, nr=n_rows, n_x=n_x, cols=theta_cols):
        dense = np.zeros((nr, n_x))
        np.add.at(dense, (jr, jc), np.asarray(prob.jacobian(x), dtype=np.float64))
        return dense[:, cols]

    J_tape = tape_jac(x0)
    ts2 = []
    for _ in range(N_REP):
        t0 = time.perf_counter()
        tape_jac(x0)
        ts2.append(time.perf_counter() - t0)
    tape_med = statistics.median(ts2)

    if J_tape.shape != J_doe.shape:
        print(f"FAIL {label}: shape {J_tape.shape} vs shipped {J_doe.shape}")
        sys.exit(1)
    err = np.max(np.abs(J_tape - J_doe) / np.maximum(1.0, np.abs(J_doe)))
    total_cmp += J_tape.size
    print(
        f"{label:16s} {doe_med * 1e3:10.1f}m {t_build * 1e3:10.2f}m "
        f"{tape_med * 1e6:10.1f}u {doe_med / (tape_med + t_build):8.0f}x "
        f"{err:12.3e}"
    )
    if err > 1e-9:
        print(f"FAIL (kill criterion) {label}: {err:.3e} > 1e-9")
        sys.exit(1)

print(
    f"\n# executed: {total_cmp} Jacobian entries compared against the shipped "
    f"FIMResult.jacobian, all within 1e-9"
)
if total_cmp == 0:
    print("FAIL: compared nothing")
    sys.exit(1)
print("PASS")
