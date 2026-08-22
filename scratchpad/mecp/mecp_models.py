"""Model potential energy surfaces for MECP / MECI optimization experiments.

Design rule for this file: every energy surface is written **once**, in terms
of ``+ - * ** exp`` and an injected ``exp``.  Passing ``numpy.exp`` evaluates
it numerically (oracle grids, local-solver baselines); passing
``discopt.modeling.exp`` builds the expression DAG.  There is deliberately no
second implementation to drift out of sync -- an earlier version of this file
had one, and a silent divergence between the two would have invalidated every
comparison in this directory.

Three families:

1. ``LVCParams`` -- linear vibronic coupling (Koeppel/Domcke/Cederbaum): two
   harmonic diabats coupled linearly through a symmetry-breaking mode.  A
   genuine conical intersection whose MECI is known in closed form, so it
   serves as the *oracle* for correctness.

2. ``MorseParams`` -- two sums of Morse oscillators in bond-length
   coordinates with different depths / equilibrium lengths: a spin-crossing
   (singlet/triplet) model.  States of different multiplicity have no
   interstate coupling, so the adiabats *are* the diabats and the MECP
   problem ``min W1 s.t. W1 = W2`` is exact rather than a surrogate.

3. ``TwoWellParams`` -- the same spin-crossing setup but with a double-well
   coordinate on the lower state (an inversion / torsion / ring-pucker mode).
   The lower surface then has two conformer wells, so the crossing seam has
   two disjoint low-lying basins with *different* energies.  This is the
   family where "did you find the global MECP, not just a local one?" has a
   checkable answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# --------------------------------------------------------------------------
# 1. LVC model (same-spin conical intersection -> MECI)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LVCParams:
    """Two-state, two-mode linear vibronic coupling Hamiltonian.

    W11 = E1 + k1*qt + wt/2*qt^2 + wc/2*qc^2
    W22 = E2 + k2*qt + wt/2*qt^2 + wc/2*qc^2
    W12 = lam*qc

    Adiabats: E_pm = (W11+W22)/2 +- sqrt(((W11-W22)/2)^2 + W12^2).
    Degeneracy needs W11 == W22 *and* W12 == 0, i.e. qc == 0 and
    qt == (E2-E1)/(k1-k2), so the MECI is analytic.
    """

    E1: float = 0.0
    E2: float = 0.30
    k1: float = 0.12
    k2: float = -0.18
    wt: float = 0.020
    wc: float = 0.015
    lam: float = 0.08
    box: tuple[float, float] = (-30.0, 30.0)

    @property
    def meci_qt(self) -> float:
        return (self.E2 - self.E1) / (self.k1 - self.k2)

    @property
    def meci_energy(self) -> float:
        qt = self.meci_qt
        return self.E1 + self.k1 * qt + 0.5 * self.wt * qt**2

    def diabats(self, qt, qc):
        """(W11, W22, W12) -- pure arithmetic, works on floats or expressions."""
        common = 0.5 * self.wt * qt**2 + 0.5 * self.wc * qc**2
        return (
            self.E1 + self.k1 * qt + common,
            self.E2 + self.k2 * qt + common,
            self.lam * qc,
        )

    def adiabats(self, qt, qc, sqrt=np.sqrt):
        w11, w22, w12 = self.diabats(qt, qc)
        half = 0.5 * (w11 + w22)
        rad = sqrt((0.5 * (w11 - w22)) ** 2 + w12**2)
        return half - rad, half + rad


# --------------------------------------------------------------------------
# 2/3. Spin-crossing models (different multiplicity => W12 == 0)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MorseParams:
    """Two sums of ``n`` Morse oscillators, one per bond-length coordinate.

    W_s(r) = dE_s + sum_i D_s*(1 - exp(-a_s*(r_i-b_s)))^2
                  + sum_{i<j} c_s*(r_i-b_s)*(r_j-b_s)
                  + tilt . r

    ``tilt`` is added to *both* states, so it cancels in the crossing
    condition W1-W2 = 0 while breaking the permutation symmetry of the energy
    *along* the seam.
    """

    n: int = 2
    D1: float = 4.0
    a1: float = 1.6
    b1: float = 1.10
    c1: float = 0.55
    D2: float = 2.6
    a2: float = 1.05
    b2: float = 1.62
    c2: float = -0.30
    dE2: float = -0.55
    tilt: float = 0.09
    box: tuple[float, float] = (0.60, 3.40)
    tilt_vec: tuple[float, ...] = field(default=())

    def tilts(self) -> np.ndarray:
        if self.tilt_vec:
            v = np.asarray(self.tilt_vec, dtype=float)
            assert v.size == self.n, f"tilt_vec size {v.size} != n {self.n}"
            return v
        return self.tilt * np.array([(-1.0) ** i * 0.85**i for i in range(self.n)])

    def bounds(self):
        return [self.box] * self.n

    def _state(self, r, D, a, b, c, dE, exp):
        t = self.tilts()
        d = [r[i] - b for i in range(self.n)]
        val = dE
        for i in range(self.n):
            val = val + D * (1.0 - exp(-a * d[i])) ** 2
        for i in range(self.n):
            for j in range(i + 1, self.n):
                val = val + c * d[i] * d[j]
        for i in range(self.n):
            val = val + t[i] * r[i]
        return val

    def states(self, r, exp=np.exp):
        """(W_lower_state, W_upper_state) at coordinate list/array ``r``."""
        w1 = self._state(r, self.D1, self.a1, self.b1, self.c1, 0.0, exp)
        w2 = self._state(r, self.D2, self.a2, self.b2, self.c2, self.dE2, exp)
        return w1, w2


@dataclass(frozen=True)
class TwoWellParams:
    """Spin-crossing model whose lower state has two conformer wells.

    Coordinate 0 is a double-well mode (inversion / torsion / ring pucker);
    coordinates 1..n-1 are Morse bond stretches.

    W1 = A*(q0^2 - s^2)^2 + sum_i D1*(1-exp(-a1*(r_i-b1)))^2
         + g1*q0*(r_1-b1) + tilt.q
    W2 = dE2 + B*q0^2 + sum_i D2*(1-exp(-a2*(r_i-b2)))^2
         + g2*q0*(r_1-b2) + tilt.q

    W1 has minima near q0 = +-s; W2 has one near q0 = 0 and sits above W1
    there.  The seam therefore has a low-lying basin on each side of q0 = 0,
    and the ``tilt`` term -- identical in both states, so it cancels from the
    crossing condition -- makes the two basins energetically inequivalent.
    """

    n: int = 2  # total coordinates: q0 plus (n-1) stretches
    A: float = 0.55
    s: float = 1.35
    B: float = 0.40
    D1: float = 4.0
    a1: float = 1.6
    b1: float = 1.10
    D2: float = 2.6
    a2: float = 1.05
    b2: float = 1.62
    dE2: float = 1.10
    g1: float = 0.30
    g2: float = -0.22
    tilt: float = 0.11
    q0_box: tuple[float, float] = (-2.60, 2.60)
    r_box: tuple[float, float] = (0.60, 3.20)

    def bounds(self):
        return [self.q0_box] + [self.r_box] * (self.n - 1)

    def tilts(self) -> np.ndarray:
        # q0 gets the symmetry-breaking tilt; stretches get a small decaying one.
        t = [self.tilt]
        for i in range(1, self.n):
            t.append(0.35 * self.tilt * (-1.0) ** i * 0.8 ** (i - 1))
        return np.array(t)

    def states(self, q, exp=np.exp):
        t = self.tilts()
        q0 = q[0]
        stretches = list(q[1:])

        w1 = self.A * (q0**2 - self.s**2) ** 2
        w2 = self.dE2 + self.B * q0**2
        for i, r in enumerate(stretches):
            w1 = w1 + self.D1 * (1.0 - exp(-self.a1 * (r - self.b1))) ** 2
            w2 = w2 + self.D2 * (1.0 - exp(-self.a2 * (r - self.b2))) ** 2
        if stretches:
            w1 = w1 + self.g1 * q0 * (stretches[0] - self.b1)
            w2 = w2 + self.g2 * q0 * (stretches[0] - self.b2)
        for i in range(self.n):
            w1 = w1 + t[i] * q[i]
            w2 = w2 + t[i] * q[i]
        return w1, w2


# --------------------------------------------------------------------------
# discopt builders
# --------------------------------------------------------------------------


def build_spin_crossing_mecp(p, name=None, gap_tol: float = 0.0):
    """Spin-crossing MECP: ``min W1 s.t. W1 - W2 == 0`` over the box.

    ``gap_tol > 0`` relaxes the equality to a two-sided band |W1-W2| <= tol,
    which is how a practitioner would pose a numerically-tolerant seam.
    """
    import discopt.modeling as dm

    m = dm.Model(name or f"{type(p).__name__}_mecp_n{p.n}")
    bnds = p.bounds()
    q = [m.continuous(f"q{i}", lb=bnds[i][0], ub=bnds[i][1]) for i in range(p.n)]
    w1, w2 = p.states(q, exp=dm.exp)
    m.minimize(w1)
    if gap_tol <= 0.0:
        m.subject_to(w1 - w2 == 0)
    else:
        m.subject_to(w1 - w2 <= gap_tol)
        m.subject_to(w1 - w2 >= -gap_tol)
    return m, q


def build_lvc_meci_diabatic(p: LVCParams):
    """MECI in the diabatic representation -- smooth, no sqrt.

    min (W11+W22)/2  s.t.  W11 - W22 == 0,  W12 == 0
    """
    import discopt.modeling as dm

    m = dm.Model("lvc_meci_diabatic")
    qt = m.continuous("qt", lb=p.box[0], ub=p.box[1])
    qc = m.continuous("qc", lb=p.box[0], ub=p.box[1])
    w11, w22, w12 = p.diabats(qt, qc)
    m.minimize(0.5 * (w11 + w22))
    m.subject_to(w11 - w22 == 0)
    m.subject_to(w12 == 0)
    return m, (qt, qc)


def build_lvc_meci_adiabatic(p: LVCParams, gap_tol: float = 0.0):
    """MECI posed on the adiabats -- the form an electronic-structure code
    hands you.  ``sqrt`` of a quantity that vanishes at the solution, so the
    objective is non-differentiable exactly at the answer.

    min E_lower  s.t.  E_upper - E_lower == 0   (== 2*rad)
    """
    import discopt.modeling as dm

    m = dm.Model("lvc_meci_adiabatic")
    qt = m.continuous("qt", lb=p.box[0], ub=p.box[1])
    qc = m.continuous("qc", lb=p.box[0], ub=p.box[1])
    lo, hi = p.adiabats(qt, qc, sqrt=dm.sqrt)
    m.minimize(lo)
    if gap_tol <= 0.0:
        m.subject_to(hi - lo == 0)
    else:
        m.subject_to(hi - lo <= gap_tol)
    return m, (qt, qc)


def build_penalty_objective(p, sigma: float, alpha: float, name=None):
    """The Levine/Coe/Martinez smooth penalty objective, as a discopt model.

    F = (W1+W2)/2 + sigma * dE^2 / (|dE| + alpha),  dE = W2 - W1

    This is *unconstrained* (box only) -- the form quantum-chemistry codes
    actually minimize.  Included to test whether discopt can globally
    optimize the objective the field already uses, ``abs`` and division
    included.
    """
    import discopt.modeling as dm

    m = dm.Model(name or f"{type(p).__name__}_penalty_n{p.n}")
    bnds = p.bounds()
    q = [m.continuous(f"q{i}", lb=bnds[i][0], ub=bnds[i][1]) for i in range(p.n)]
    w1, w2 = p.states(q, exp=dm.exp)
    d = w2 - w1
    m.minimize(0.5 * (w1 + w2) + sigma * d**2 / (dm.abs(d) + alpha))
    return m, q


def penalty_objective_np(p, x, sigma: float, alpha: float) -> float:
    w1, w2 = p.states(list(np.asarray(x, dtype=float)), exp=np.exp)
    d = w2 - w1
    return float(0.5 * (w1 + w2) + sigma * d**2 / (abs(d) + alpha))


# --------------------------------------------------------------------------
# Oracles
# --------------------------------------------------------------------------


def seam_oracle_2d(p, n_grid: int = 1601, n_bisect: int = 60):
    """Brute-force the seam for a 2-coordinate model.

    Finds every grid edge where W1-W2 changes sign, bisects it to machine
    precision, and returns the lowest W1 over all of them.

    Returns ``(best_energy, best_point, n_crossings, basins)`` where
    ``basins`` lists (energy, point) for each connected low group -- used to
    show that more than one distinct local MECP exists.
    """
    assert p.n == 2, "2-D oracle only"
    (lo0, hi0), (lo1, hi1) = p.bounds()
    g0 = np.linspace(lo0, hi0, n_grid)
    g1 = np.linspace(lo1, hi1, n_grid)
    G0, G1 = np.meshgrid(g0, g1, indexing="ij")
    w1, w2 = p.states([G0, G1], exp=np.exp)
    d = w1 - w2

    pts: list[tuple[float, np.ndarray]] = []
    n_cross = 0
    for axis in (0, 1):
        if axis == 0:
            mask = np.sign(d[:-1, :]) * np.sign(d[1:, :]) < 0
        else:
            mask = np.sign(d[:, :-1]) * np.sign(d[:, 1:]) < 0
        idx = np.argwhere(mask)
        n_cross += int(idx.shape[0])
        for i, j in idx:
            a = np.array([g0[i], g1[j]])
            b = np.array([g0[i + 1], g1[j]]) if axis == 0 else np.array([g0[i], g1[j + 1]])
            fa = float(d[i, j])
            for _ in range(n_bisect):
                mid = 0.5 * (a + b)
                wm1, wm2 = p.states(list(mid), exp=np.exp)
                fm = float(wm1 - wm2)
                if fa * fm <= 0:
                    b = mid
                else:
                    a, fa = mid, fm
            mid = 0.5 * (a + b)
            e = float(p.states(list(mid), exp=np.exp)[0])
            pts.append((e, mid))

    assert n_cross > 0, "oracle found no crossing -- model has no seam in this box"
    pts.sort(key=lambda t: t[0])

    # Group the crossing points into basins: greedily take the lowest point,
    # then discard everything within ``radius`` of it, repeat.
    radius = 0.25 * min(hi0 - lo0, hi1 - lo1)
    basins: list[tuple[float, np.ndarray]] = []
    remaining = list(pts)
    while remaining and len(basins) < 12:
        e, x = remaining[0]
        basins.append((e, x))
        remaining = [(e2, x2) for e2, x2 in remaining if np.linalg.norm(x2 - x) > radius]

    return pts[0][0], pts[0][1], n_cross, basins


def seam_oracle_sampled(p, n_samples: int = 400_000, seed: int = 0, n_refine: int = 400):
    """Randomised seam oracle for n > 2: sample the box, keep the points with
    the smallest |W1-W2|, project each onto the seam by a 1-D Newton step
    along grad(W1-W2), and report the lowest W1 found.

    This is a *stochastic upper bound* on the true MECP energy, not a
    certificate.  Reported as such.
    """
    rng = np.random.default_rng(seed)
    bnds = np.array(p.bounds(), dtype=float)
    X = rng.uniform(bnds[:, 0], bnds[:, 1], size=(n_samples, p.n))
    w1, w2 = p.states([X[:, i] for i in range(p.n)], exp=np.exp)
    gap = np.abs(w1 - w2)
    order = np.argsort(w1 + 50.0 * gap)  # favour low energy AND small gap
    best = (np.inf, None)
    n_proj = 0

    def constraint(x):
        a, b = p.states(list(x), exp=np.exp)
        return float(a - b)

    def grad_constraint(x, h=1e-6):
        g = np.zeros(p.n)
        for i in range(p.n):
            xp, xm = x.copy(), x.copy()
            xp[i] += h
            xm[i] -= h
            g[i] = (constraint(xp) - constraint(xm)) / (2 * h)
        return g

    for k in order[:n_refine]:
        x = X[k].copy()
        ok = False
        for _ in range(80):
            c = constraint(x)
            if abs(c) < 1e-10:
                ok = True
                break
            g = grad_constraint(x)
            gg = float(g @ g)
            if gg < 1e-18:
                break
            x = x - (c / gg) * g
            x = np.clip(x, bnds[:, 0], bnds[:, 1])
        if not ok or abs(constraint(x)) > 1e-7:
            continue
        n_proj += 1
        e = float(p.states(list(x), exp=np.exp)[0])
        if e < best[0]:
            best = (e, x.copy())
    assert n_proj > 0, "sampled oracle projected nothing onto the seam"
    return best[0], best[1], n_proj


if __name__ == "__main__":
    checks = 0
    rng = np.random.default_rng(0)

    lp = LVCParams()
    lo, hi = lp.adiabats(lp.meci_qt, 0.0)
    assert abs(hi - lo) < 1e-12, f"LVC degeneracy failed: gap={hi - lo}"
    assert abs(lo - lp.meci_energy) < 1e-12
    checks += 2
    print(f"LVC analytic MECI: qt={lp.meci_qt:.10f} qc=0 E={lp.meci_energy:.10f}")

    for _ in range(200):
        q = rng.uniform(-20, 20, size=2)
        w11, w22, _ = lp.diabats(*q)
        lo, hi = lp.adiabats(*q)
        assert lo <= min(w11, w22) + 1e-12 and hi >= max(w11, w22) - 1e-12
        checks += 1

    mp = MorseParams(n=2)
    e, x, ncross, basins = seam_oracle_2d(mp, n_grid=801)
    print(f"\nMorse n=2 : E_MECP={e:.8f} at {np.round(x, 5)}  crossings={ncross}")
    for i, (be, bx) in enumerate(basins[:5]):
        print(f"   basin {i}: E={be:.6f} at {np.round(bx, 4)}")
    checks += 1

    tw = TwoWellParams(n=2)
    e2, x2, ncross2, basins2 = seam_oracle_2d(tw, n_grid=801)
    print(f"\nTwoWell n=2 : E_MECP={e2:.8f} at {np.round(x2, 5)}  crossings={ncross2}")
    for i, (be, bx) in enumerate(basins2[:6]):
        print(f"   basin {i}: E={be:.6f} at {np.round(bx, 4)}")
    checks += 1

    for n in (3, 4):
        twn = TwoWellParams(n=n)
        es, xs, npj = seam_oracle_sampled(twn, n_samples=200_000, n_refine=300)
        print(f"\nTwoWell n={n}: sampled seam best E={es:.6f} at {np.round(xs, 4)} (proj={npj})")
        checks += 1

    print(f"\nEXECUTED ASSERTIONS: {checks}")
    if checks == 0:
        raise SystemExit(1)
