#!/usr/bin/env python
"""Can POUNCE's Rust tape supply mb-doe's dY/dtheta, replacing jax.jacrev? (#1215)

mb-doe's FIM needs J[i,j] = dy_i/dtheta_j. discopt-doe gets it from
``jax.jacrev`` over a JAX closure compiled from the Python DAG, and the
mb-doe profile puts that path at 73-92% of total cost (the XLA trace plus the
reverse-mode evaluation).

The tape already does exactly this shape for Gauss-Newton: "tape the residual
vector r(x) as the constraint rows of an auxiliary NlProblem and read its R x n
Jacobian". Responses are residuals under a different name, so this taping the
RESPONSES as constraint rows of an aux problem and reading the theta columns of
its Jacobian.

Kill criterion: if the tape Jacobian does not match jax.jacrev to 1e-9, this
direction is dead and the JAX path stays.

Measured 2026-09-09 on `main` @ c052e85 (4 cores, load <= 0.2), exact agreement
(max relative error 0.000e+00) at every size:

============  ============  ===========  ===========  ==========
N_EXP / NFE   tape build    jax build    tape eval    jax eval
============  ============  ===========  ===========  ==========
8 / 6              0.41 ms     116.18 ms      5.6 us     10.9 us
32 / 8             0.91 ms     186.73 ms     16.7 us     47.9 us
64 / 12            2.40 ms     300.74 ms    158.7 us     73.9 us
============  ============  ===========  ===========  ==========

The tape's setup is 50-125x cheaper because it has no XLA trace to pay. Per
evaluation it wins at the two smaller sizes and loses ~2x at the largest, with a
large spread (sd 527 us) that is not explained -- do not quote the 64/12 eval
column as settled.

§6: prints the number of Jacobian entries compared; exits non-zero if 0 or if
any entry disagrees. The 1e-9 agreement bar is a stated kill criterion, not a
post-hoc threshold.
§8: prints the module file it loaded.
§9: reports a load gate; both arms are timed over 200 calls with a spread.

Usage::

    python -u discopt_benchmarks/scripts/issue1215_tape_sensitivity.py
    N_EXP=64 NFE=12 python -u discopt_benchmarks/scripts/issue1215_tape_sensitivity.py
"""

import os
import statistics
import sys
import time

import discopt.modeling as dm
import discopt.modeling.core as core
import numpy as np
from discopt.modeling import Model

N_EXP = int(os.environ.get("N_EXP", 8))
N_THETA = 4
NFE, NCP = int(os.environ.get("NFE", 6)), 3

print(f"# discopt loaded from: {core.__file__}")
print(f"# load: {os.getloadavg()[0]:.2f}   N_EXP={N_EXP} NFE={NFE}")


def build():
    from discopt.dae import ContinuousSet, DAEBuilder

    m = Model("mbdoe")
    theta = m.continuous("theta", shape=(N_THETA,), lb=0.01, ub=5.0)
    designs = m.continuous("d", shape=(N_EXP,), lb=0.1, ub=2.0)
    responses = []
    for e in range(N_EXP):
        dae = DAEBuilder(m, ContinuousSet(f"t{e}", bounds=(0, 2), nfe=NFE, ncp=NCP))
        dae.add_state(f"A{e}", initial=1.0, bounds=(1e-6, 10.0))
        dae.add_state(f"B{e}", initial=0.0, bounds=(0.0, 10.0))
        dae.set_ode(
            lambda t, s, a, c, e=e: {
                f"A{e}": -theta[0] * dm.exp(-theta[1] / (1.0 + designs[e])) * s[f"A{e}"] ** 2,
                f"B{e}": theta[2] * s[f"A{e}"] - theta[3] * s[f"B{e}"],
            }
        )
        dae.discretize()
        st = next(v for v in m._variables if v.name == f"t{e}_B{e}")
        responses.append(st[NFE - 1, NCP])
    m.minimize(theta[0])
    return m, responses


m, responses = build()
offs, o = {}, 0
for v in m._variables:
    offs[v.name] = o
    o += int(v.size)
n_x = o
t_off, d_off = offs["theta"], offs["d"]
theta_cols = list(range(t_off, t_off + N_THETA))

rng = np.random.default_rng(0)
lo = np.concatenate([np.asarray(v.lb).reshape(-1) for v in m._variables])
hi = np.concatenate([np.asarray(v.ub).reshape(-1) for v in m._variables])
x0 = np.clip(lo + 0.37 * (np.minimum(hi, 10.0) - lo), lo, hi)

# ── arm A: the tape, via an aux NlProblem over the response rows ──
import pounce  # noqa: E402
from discopt._nl_expr_compiler import compile_to_nl_expr  # noqa: E402

t0 = time.perf_counter()
resp_exprs = [compile_to_nl_expr(r, m) for r in responses]
prob = pounce.build_nl_problem(
    n_x, pounce.NlExpr.const_(0.0), constraints=resp_exprs, x_l=lo, x_u=hi
)
t_tape_build = time.perf_counter() - t0

jr, jc = (np.asarray(a) for a in prob.jacobian_structure())


def tape_jac(x):
    vals = np.asarray(prob.jacobian(x), dtype=np.float64)
    dense = np.zeros((len(responses), n_x))
    np.add.at(dense, (jr, jc), vals)
    return dense[:, theta_cols]


J_tape = tape_jac(x0)
print(f"A. tape  build {t_tape_build * 1e3:8.2f} ms   J {J_tape.shape}")

# ── arm B: jax.jacrev, what discopt-doe does today ──
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from discopt._relax.dag_compiler import compile_expression  # noqa: E402

t0 = time.perf_counter()
fns = [compile_expression(r, m) for r in responses]


def y_all(theta_vec, rest):
    x = jnp.asarray(rest)
    x = x.at[t_off : t_off + N_THETA].set(theta_vec)
    return jnp.stack([f(x) for f in fns])


jac = jax.jit(jax.jacrev(y_all, argnums=0))
J_jax = np.asarray(jac(x0[t_off : t_off + N_THETA], x0))
t_jax_build = time.perf_counter() - t0
print(f"B. jax   build {t_jax_build * 1e3:8.2f} ms   J {J_jax.shape}  (incl. XLA trace)")

# ── agreement (the kill criterion) ──
if J_tape.shape != J_jax.shape:
    print(f"FAIL: shape mismatch {J_tape.shape} vs {J_jax.shape}")
    sys.exit(1)
n_cmp = J_tape.size
scale = np.maximum(1.0, np.abs(J_jax))
err = np.max(np.abs(J_tape - J_jax) / scale)
print(f"\n# compared {n_cmp} Jacobian entries, max relative err {err:.3e}")
if n_cmp == 0:
    print("FAIL: compared nothing")
    sys.exit(1)
if err > 1e-9:
    print(f"FAIL (kill criterion): tape disagrees with jax.jacrev at {err:.3e} > 1e-9")
    worst = np.unravel_index(np.argmax(np.abs(J_tape - J_jax) / scale), J_tape.shape)
    print(f"  worst entry {worst}: tape {J_tape[worst]!r} vs jax {J_jax[worst]!r}")
    sys.exit(1)
print("PASS: the tape reproduces dY/dtheta")

# ── steady-state evaluation cost ──
N = 200
for label, fn in (
    ("tape", lambda: tape_jac(x0)),
    ("jax ", lambda: np.asarray(jac(x0[t_off : t_off + N_THETA], x0))),
):
    ts = []
    for _ in range(N):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    print(
        f"{label} eval: median {statistics.median(ts) * 1e6:9.1f} us  "
        f"sd {statistics.stdev(ts) * 1e6:8.1f}  over {N} calls"
    )
print(f"\n# executed: {n_cmp} entries compared, {2 * N} evaluations timed")
