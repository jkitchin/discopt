"""EXPERIMENT 4 -- coordinate choice: Cartesian vs internal.

Every production MECP code works in 3N Cartesian coordinates and projects out
the 6 translational/rotational degrees of freedom.  For a *local* optimizer
that redundancy is harmless -- it just means the Hessian has 6 zero modes.
For a *global* branch-and-bound solver it is potentially fatal: the set of
global optima is a 6-dimensional manifold (every rotation/translation of the
MECP is also a MECP), so no amount of branching ever isolates a point, and
the solver cannot prune boxes that all contain an optimum.

This experiment measures that.  A 3-atom system, two states, with energies
that depend only on the three interatomic distances:

  A. internal   -- 3 variables (the squared bond distances), no redundancy
  B. cartesian  -- 9 variables, energies built from Cartesian differences,
                   6-fold redundancy left in
  C. cartesian + anchoring -- 9 variables with atom 1 fixed at the origin,
                   atom 2 on the +x axis, atom 3 in the xy plane with y >= 0
                   (the standard 3N-6 gauge fixing), which removes it again

If (B) blows up and (C) recovers (A)'s behaviour, then the practical answer
is "discopt needs gauge-fixed or internal coordinates", which is a concrete,
documentable requirement rather than a vague limitation.
"""

from __future__ import annotations

import sys
import time

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])

import discopt.modeling as dm  # noqa: E402

CHECKS = 0
NOTES: list[str] = []
TIME_LIMIT = float(sys.argv[1]) if len(sys.argv) > 1 else 180.0

# Two Morse-in-distance states over the three bonds of a triatomic.
# State 1 (e.g. singlet): deeper, shorter bonds. State 2 (triplet): shallower,
# longer, offset.  Written in terms of the *distance* d (not d^2) so both
# formulations evaluate the identical energy function.
P1 = dict(D=4.0, a=1.55, b=1.15, dE=0.0)
P2 = dict(D=2.5, a=1.00, b=1.66, dE=-0.40)
TILT = np.array([0.10, -0.06, 0.03])  # same in both states: cancels in the seam
D_LO, D_HI = 0.70, 3.20


def state_energy(dists, p, exp):
    val = p["dE"]
    for k, d in enumerate(dists):
        val = val + p["D"] * (1.0 - exp(-p["a"] * (d - p["b"]))) ** 2 + TILT[k] * d
    return val


def both(dists, exp):
    return state_energy(dists, P1, exp), state_energy(dists, P2, exp)


def run(tag, build, time_limit=TIME_LIMIT):
    global CHECKS
    m, extract = build()
    n_var = len(m._variables) if hasattr(m, "_variables") else -1
    t0 = time.time()
    r = m.solve(time_limit=time_limit)
    el = time.time() - t0
    print(
        f"  {tag:<26s} nvar={n_var:<4d} status={r.status:<11s} "
        f"obj={_n(r.objective)} bound={_n(r.bound)} gap={_g(r.gap)} "
        f"cert={str(r.gap_certified):<5s} nodes={r.node_count:<7d} t={el:7.2f}s"
    )
    CHECKS += 1
    dists = None
    if r.x is not None:
        try:
            dists = extract(r.x)
            w1, w2 = both(dists, np.exp)
            print(
                f"      recovered bonds = {np.round(np.sort(dists), 5)}  "
                f"W1={float(w1):.6f}  |W1-W2|={abs(float(w1 - w2)):.2e}"
            )
            CHECKS += 1
        except Exception as exc:
            NOTES.append(f"{tag}: could not recover distances: {type(exc).__name__}: {exc}")
    return r, dists


def _n(v):
    return "     None" if v is None else f"{v:10.6f}"


def _g(v):
    return "    None" if v is None else f"{v:8.1e}"


# ==========================================================================
print("=" * 90)
print(f"EXPERIMENT 4 -- coordinate systems for a triatomic MECP (limit {TIME_LIMIT}s)")
print("=" * 90)


# --- A: internal coordinates (the three bond distances) --------------------
def build_internal():
    m = dm.Model("mecp_internal")
    d = [m.continuous(f"d{k}", lb=D_LO, ub=D_HI) for k in range(3)]
    w1, w2 = both(d, dm.exp)
    m.minimize(w1)
    m.subject_to(w1 - w2 == 0)
    return m, (lambda x: np.array([float(np.asarray(x[f"d{k}"]).ravel()[0]) for k in range(3)]))


# --- B/C: Cartesian coordinates -------------------------------------------
def _cart_build(anchor: bool):
    m = dm.Model("mecp_cart_anchored" if anchor else "mecp_cart_free")
    # Box big enough to contain a triangle of the right size, plus slack.
    LO, HI = -4.0, 4.0
    coords = {}
    for i in range(3):
        for c, cname in enumerate("xyz"):
            if anchor:
                # gauge: atom0 at origin; atom1 on +x; atom2 in xy plane, y>=0
                if i == 0:
                    lo = hi = 0.0
                elif i == 1:
                    lo, hi = (D_LO, D_HI) if c == 0 else (0.0, 0.0)
                else:
                    lo, hi = (LO, HI) if c == 0 else ((0.0, HI) if c == 1 else (0.0, 0.0))
            else:
                lo, hi = LO, HI
            if lo == hi:
                coords[(i, cname)] = lo  # a fixed constant, not a variable
            else:
                coords[(i, cname)] = m.continuous(f"{cname}{i}", lb=lo, ub=hi)

    def dist(i, j):
        s = 0.0
        for cname in "xyz":
            diff = coords[(i, cname)] - coords[(j, cname)]
            s = s + diff * diff
        return dm.sqrt(s)

    d = [dist(0, 1), dist(0, 2), dist(1, 2)]
    w1, w2 = both(d, dm.exp)
    m.minimize(w1)
    m.subject_to(w1 - w2 == 0)
    # keep the atoms apart so sqrt stays away from 0 (a real modelling need)
    for i in range(3):
        for j in range(i + 1, 3):
            s = 0.0
            for cname in "xyz":
                diff = coords[(i, cname)] - coords[(j, cname)]
                s = s + diff * diff
            m.subject_to(s >= D_LO**2)
            m.subject_to(s <= D_HI**2)

    def extract(x):
        pos = np.zeros((3, 3))
        for i in range(3):
            for c, cname in enumerate("xyz"):
                v = coords[(i, cname)]
                pos[i, c] = (
                    v if isinstance(v, float) else float(np.asarray(x[f"{cname}{i}"]).ravel()[0])
                )
        return np.array(
            [
                np.linalg.norm(pos[0] - pos[1]),
                np.linalg.norm(pos[0] - pos[2]),
                np.linalg.norm(pos[1] - pos[2]),
            ]
        )

    return m, extract


print("\n[A] internal coordinates (3 bond distances, no redundancy)")
rA, dA = run("A internal", build_internal)

print("\n[B] free Cartesian coordinates (9 vars, 6 redundant DOF)")
rB, dB = run("B cartesian free", lambda: _cart_build(anchor=False))

print("\n[C] gauge-fixed Cartesian (Eckart-style anchoring)")
rC, dC = run("C cartesian anchored", lambda: _cart_build(anchor=True))

# ==========================================================================
print("\n" + "=" * 90)
print("COMPARISON")
for tag, r in (("A internal", rA), ("B cart free", rB), ("C cart anchored", rC)):
    print(
        f"  {tag:<18s} cert={str(r.gap_certified):<6s} nodes={r.node_count:<8d} "
        f"obj={_n(r.objective)} status={r.status}"
    )
if rA.objective is not None:
    for tag, r, d in (("B", rB, dB), ("C", rC, dC)):
        CHECKS += 1
        if r.objective is None:
            NOTES.append(f"{tag} found no solution while A did")
        elif abs(r.objective - rA.objective) > 1e-3:
            NOTES.append(
                f"{tag} objective {r.objective:.6f} != internal-coordinate "
                f"answer {rA.objective:.6f} (same physical problem)"
            )
    if rB.node_count > 20 * max(rA.node_count, 1):
        print(
            f"\n  => free-Cartesian redundancy costs >20x the nodes "
            f"({rB.node_count} vs {rA.node_count})"
        )

print(f"\nEXECUTED CHECKS: {CHECKS}")
print(f"NOTES: {len(NOTES)}")
for s in NOTES:
    print(f"  - {s}")
if CHECKS == 0:
    sys.exit(2)
