"""EXPERIMENT 3b -- retest of the dm.custom reduced-space route.

Experiment 3 part B reported "no certificate" for both variants, but the probe
was wrong: it passed the whole coordinate *vector* as a single CustomCall
argument, and a non-scalar leaf is one of the documented disqualifiers for the
MCBox reduced-space path (see docs/notebooks/reduced_space_customcall.md).
The measurement therefore never exercised the route it claimed to test.

This retest passes scalar components, as the documented pattern requires, and
compares four things on the identical physical problem:

  1. factorable   -- ordinary dm.exp expressions (the §2.1 baseline)
  2. custom/raw   -- dm.custom with scalar args and a raw jnp.exp
  3. custom/mcbox -- dm.custom with scalar args and an MCBox-dispatching exp
  4. udf          -- dm.udf, a symbolic body built from dm.* primitives

Expected, if the documentation is accurate: (1) and (4) certify; (3) certifies
while branching only on the geometry DOF; (2) does not certify.
"""

from __future__ import annotations

import sys
import time

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])

import discopt.modeling as dm  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import mecp_models as M  # noqa: E402
from discopt._relax.mcbox import MCBox  # noqa: E402

CHECKS = 0
NOTES: list[str] = []
NDIM = int(sys.argv[1]) if len(sys.argv) > 1 else 3
TL = float(sys.argv[2]) if len(sys.argv) > 2 else 300.0

tw = M.TwoWellParams(n=NDIM)
BNDS = np.array(tw.bounds(), float)

# Reference: the certified answer from exp2 for this exact model/dimension,
# recomputed here so this script stands alone.
E_REF = None


def mexp(x):
    """Dispatching exp: MCBox on the relaxation path, jnp on the value path."""
    return x.exp() if isinstance(x, MCBox) else jnp.exp(x)


def show(tag, r, el):
    print(
        f"  {tag:<22s} status={r.status:<11s} obj={_n(r.objective)} bound={_n(r.bound)} "
        f"gap={_g(r.gap)} cert={str(r.gap_certified):<5s} nodes={r.node_count:<7d} t={el:7.2f}s"
    )


def _n(v):
    return "     None" if v is None else f"{v:10.6f}"


def _g(v):
    return "    None" if v is None else f"{v:8.1e}"


def score(tag, x):
    w1, w2 = tw.states(list(np.asarray(x, float)), exp=np.exp)
    print(f"     -> true W1={float(w1):.6f}  |W1-W2|={abs(float(w1 - w2)):.2e}")
    return float(w1)


print("=" * 96)
print(f"EXPERIMENT 3b -- reduced-space retest, n={NDIM}, limit {TL}s")
print("=" * 96)

results = {}

# --- 1. factorable baseline ------------------------------------------------
m, q = M.build_spin_crossing_mecp(tw)
t0 = time.time()
r = m.solve(time_limit=TL)
el = time.time() - t0
show("1 factorable", r, el)
E_REF = r.objective
CHECKS += 1
results["factorable"] = (r, el)
if r.x is not None:
    score("factorable", [float(np.asarray(r.x[f"q{i}"]).ravel()[0]) for i in range(NDIM)])

# --- 2/3. dm.custom with SCALAR arguments ---------------------------------
for tag, expfn in (("2 custom/raw-jnp", jnp.exp), ("3 custom/mcbox", mexp)):
    try:

        def w1s(*args, _e=expfn):
            return tw.states(list(args), exp=_e)[0]

        def w2s(*args, _e=expfn):
            return tw.states(list(args), exp=_e)[1]

        m = dm.Model(f"custom_scalar_{tag.split('/')[-1]}")
        qs = [m.continuous(f"q{i}", lb=BNDS[i, 0], ub=BNDS[i, 1]) for i in range(NDIM)]
        c1 = dm.custom(w1s, name="W1")
        c2 = dm.custom(w2s, name="W2")
        m.minimize(c1(*qs))
        m.subject_to(c1(*qs) - c2(*qs) == 0)
        t0 = time.time()
        r = m.solve(time_limit=TL)
        el = time.time() - t0
        show(tag, r, el)
        CHECKS += 1
        results[tag] = (r, el)
        if r.x is not None:
            score(tag, [float(np.asarray(r.x[f"q{i}"]).ravel()[0]) for i in range(NDIM)])
        if r.gap_certified and r.bound is not None and E_REF is not None:
            CHECKS += 1
            if r.bound > E_REF + 1e-4:
                NOTES.append(f"SOUNDNESS: {tag} bound {r.bound} > factorable optimum {E_REF}")
    except Exception as exc:
        print(f"  {tag}: RAISED {type(exc).__name__}: {exc}")
        NOTES.append(f"{tag} raised {type(exc).__name__}: {exc}")
        CHECKS += 1

# --- 4. dm.udf, symbolic body --------------------------------------------
try:
    m = dm.Model("udf_mecp")
    qs = [m.continuous(f"q{i}", lb=BNDS[i, 0], ub=BNDS[i, 1]) for i in range(NDIM)]
    u1 = dm.udf(lambda *a: tw.states(list(a), exp=dm.exp)[0])
    u2 = dm.udf(lambda *a: tw.states(list(a), exp=dm.exp)[1])
    m.minimize(u1(*qs))
    m.subject_to(u1(*qs) - u2(*qs) == 0)
    t0 = time.time()
    r = m.solve(time_limit=TL)
    el = time.time() - t0
    show("4 udf/symbolic", r, el)
    CHECKS += 1
    results["udf"] = (r, el)
    if r.x is not None:
        score("udf", [float(np.asarray(r.x[f"q{i}"]).ravel()[0]) for i in range(NDIM)])
except Exception as exc:
    print(f"  4 udf: RAISED {type(exc).__name__}: {exc}")
    NOTES.append(f"udf raised {type(exc).__name__}: {exc}")
    CHECKS += 1

print("\n" + "=" * 96)
print(f"EXECUTED CHECKS: {CHECKS}")
print(f"NOTES: {len(NOTES)}")
for s in NOTES:
    print(f"  - {s}")
if CHECKS == 0:
    sys.exit(2)
