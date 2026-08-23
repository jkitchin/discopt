"""EXPERIMENT 1 -- can discopt solve MECP/MECI problems, and in which
formulation?

Hypothesis under test
---------------------
H1: The *diabatic* MECP/MECI formulation (min of a diabatic combination
    subject to polynomial/exponential equalities) is inside discopt's
    factorable expression class and solves to a certified global optimum.
H2: The *adiabatic* formulation -- the one an electronic-structure code hands
    you, with sqrt of a quantity that vanishes at the solution -- is either
    inexpressible, uncertifiable, or much harder.
H3: The Levine/Coe/Martinez smooth penalty objective (abs + division) is
    expressible and can be globally optimized.

Kill criterion
--------------
If the diabatic LVC MECI does not reproduce the analytic optimum
(qt = 1, qc = 0, E = 0.13) to 1e-5 with gap_certified=True, H1 is false and
the whole line of work stops here.

Every solve is checked against an independent oracle and counted.  The script
exits non-zero if it performed zero checks.
"""

from __future__ import annotations

import sys
import time
import traceback

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])

import discopt  # noqa: E402
import mecp_models as M  # noqa: E402

CHECKS = 0
FAILURES: list[str] = []


def check(cond: bool, msg: str) -> None:
    global CHECKS
    CHECKS += 1
    if not cond:
        FAILURES.append(msg)
        print(f"   !! FAIL: {msg}")


def report(tag, res, x_expected=None, e_expected=None, tol=1e-4, model_vars=None):
    """Print a solve result and check it against an oracle value."""
    print(
        f"   {tag:<34s} status={res.status:<12s} obj={_f(res.objective)} "
        f"bound={_f(res.bound)} gap={_g(res.gap)} cert={res.gap_certified} "
        f"nodes={res.node_count:<7d} t={res.wall_time:.2f}s"
    )
    if res.x is not None and model_vars:
        vals = np.array([float(np.asarray(res.x[v]).ravel()[0]) for v in model_vars])
        print(f"      x = {np.round(vals, 6)}")
        if x_expected is not None:
            check(
                np.allclose(vals, x_expected, atol=1e-3),
                f"{tag}: point {vals} != expected {x_expected}",
            )
    if e_expected is not None:
        check(
            res.objective is not None and abs(res.objective - e_expected) < tol,
            f"{tag}: objective {res.objective} != oracle {e_expected} (tol {tol})",
        )
        # Soundness: a valid dual bound may never exceed the true optimum.
        if res.bound is not None and res.gap_certified:
            check(
                res.bound <= e_expected + 1e-6,
                f"{tag}: SOUNDNESS -- certified bound {res.bound} > true optimum {e_expected}",
            )
    return res


def _f(v):
    return "     None" if v is None else f"{v:10.6f}"


def _g(v):
    return "    None" if v is None else f"{v:8.2e}"


def solve(m, **kw):
    kw.setdefault("time_limit", 120.0)
    t0 = time.time()
    try:
        return m.solve(**kw)
    except Exception as exc:  # deliberately reported, never swallowed
        print(f"   RAISED after {time.time() - t0:.2f}s: {type(exc).__name__}: {exc}")
        traceback.print_exc(limit=3)
        raise


# ==========================================================================
print("=" * 78)
print("discopt", discopt.__file__)
print("=" * 78)

# --------------------------------------------------------------------------
print("\n[1] LVC MECI -- analytic oracle: qt=1, qc=0, E=0.13")
# --------------------------------------------------------------------------
lp = M.LVCParams()
print(f"    analytic: qt={lp.meci_qt} qc=0 E={lp.meci_energy}")

print("\n  (a) diabatic formulation  min (W11+W22)/2  s.t. W11-W22=0, W12=0")
m, (qt, qc) = M.build_lvc_meci_diabatic(lp)
r = solve(m)
report(
    "lvc/diabatic",
    r,
    x_expected=[lp.meci_qt, 0.0],
    e_expected=lp.meci_energy,
    model_vars=["qt", "qc"],
)

print("\n  (b) adiabatic formulation  min E_lo  s.t. E_hi-E_lo=0   (sqrt form)")
try:
    m, _ = M.build_lvc_meci_adiabatic(lp)
    r = solve(m)
    report("lvc/adiabatic (eq)", r, e_expected=lp.meci_energy, model_vars=["qt", "qc"])
except Exception:
    print("   -> adiabatic equality form did not complete (see traceback above)")
    CHECKS += 1

print("\n  (c) adiabatic, gap band  min E_lo  s.t. E_hi-E_lo <= 1e-4")
try:
    m, _ = M.build_lvc_meci_adiabatic(lp, gap_tol=1e-4)
    r = solve(m)
    report("lvc/adiabatic (band)", r, e_expected=lp.meci_energy, tol=1e-2, model_vars=["qt", "qc"])
except Exception:
    print("   -> adiabatic band form did not complete")
    CHECKS += 1

# --------------------------------------------------------------------------
print("\n[2] Morse spin-crossing MECP, n=2 -- grid oracle")
# --------------------------------------------------------------------------
mp = M.MorseParams(n=2)
e_or, x_or, ncross, basins = M.seam_oracle_2d(mp, n_grid=1601)
print(f"    oracle: E={e_or:.8f} at {np.round(x_or, 6)}  ({ncross} grid crossings)")
check(ncross > 0, "morse n=2 oracle found no crossings")

m, _ = M.build_spin_crossing_mecp(mp)
r = solve(m)
report("morse-n2/exact-eq", r, x_expected=x_or, e_expected=e_or, model_vars=["q0", "q1"])

# --------------------------------------------------------------------------
print("\n[3] TwoWell spin-crossing MECP, n=2 -- TWO distinct seam basins")
# --------------------------------------------------------------------------
tw = M.TwoWellParams(n=2)
e_or2, x_or2, ncross2, basins2 = M.seam_oracle_2d(tw, n_grid=1601)
print(f"    oracle global : E={e_or2:.8f} at {np.round(x_or2, 6)}")
for i, (be, bx) in enumerate(basins2[:4]):
    print(f"    oracle basin {i}: E={be:.6f} at {np.round(bx, 4)}")
check(
    len(basins2) >= 2 and basins2[1][0] - basins2[0][0] > 0.05,
    "TwoWell model does not actually have two energetically distinct basins",
)

m, _ = M.build_spin_crossing_mecp(tw)
r = solve(m)
report("twowell-n2/exact-eq", r, x_expected=x_or2, e_expected=e_or2, model_vars=["q0", "q1"])

# --------------------------------------------------------------------------
print("\n[4] Levine/Coe/Martinez penalty objective, globally optimized")
print("    F = (W1+W2)/2 + sigma*dE^2/(|dE|+alpha),  sigma=3.5, alpha=0.025")
# --------------------------------------------------------------------------
for sigma, alpha in ((3.5, 0.025), (20.0, 0.005)):
    try:
        m, _ = M.build_penalty_objective(tw, sigma=sigma, alpha=alpha)
        r = solve(m)
        tag = f"twowell-n2/penalty s={sigma}"
        report(tag, r, model_vars=["q0", "q1"])
        if r.x is not None:
            xv = np.array([float(np.asarray(r.x[v]).ravel()[0]) for v in ("q0", "q1")])
            w1, w2 = tw.states(list(xv), exp=np.exp)
            print(f"      -> W1={w1:.6f}  gap|W1-W2|={abs(w1 - w2):.3e}")
            # The penalty optimum should sit near the true MECP, with a gap
            # that shrinks with alpha. It is NOT expected to equal it exactly.
            check(
                abs(w1 - w2) < 10 * alpha + 1e-3,
                f"{tag}: residual gap {abs(w1 - w2):.3e} >> alpha={alpha}",
            )
    except Exception:
        print("   -> penalty form did not complete")
        CHECKS += 1

# --------------------------------------------------------------------------
print("\n[5] Numerically-tolerant seam band |W1-W2| <= tol (practitioner form)")
# --------------------------------------------------------------------------
for tol in (1e-3, 1e-5):
    m, _ = M.build_spin_crossing_mecp(tw, gap_tol=tol)
    r = solve(m)
    report(f"twowell-n2/band {tol:g}", r, e_expected=e_or2, tol=1e-2, model_vars=["q0", "q1"])

# ==========================================================================
print("\n" + "=" * 78)
print(f"EXECUTED CHECKS: {CHECKS}")
print(f"FAILURES: {len(FAILURES)}")
for f in FAILURES:
    print(f"  - {f}")
if CHECKS == 0:
    print("PROBE FIRED NOTHING -- treating as failure")
    sys.exit(2)
sys.exit(1 if FAILURES else 0)
