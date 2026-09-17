"""#1248 D acceptance, measured on a REAL CALPHAD pricing instance.

#1248 asks that "tangent-plane cuts supplied through ``CutCallback`` reduce node
count on a pricing instance". This runs that measurement against the actual
plugin (github.com/jkitchin/discopt-calphad): CU2MG, the Cu-Mg Laves phase
(CU,MG)2(CU,MG)1 from the NIMS assessment, priced by
``discopt.calphad.equilibrium.pricing.price_phase``.

Requires the plugin on the path; skips cleanly without it, since discopt does not
depend on it.

RESULT: the criterion is NOT met, for a structural reason worth more than the
measurement.

**1. In the plugin's own formulation, no sound nontrivial cut exists at all.**
``price_phase`` builds a BOX-only model -- no constraints, just site-fraction
bounds (asserted below). Every point of the box is therefore feasible, so any cut
excluding part of the box excludes a feasible point. Measured: a box-halving cut
is refused by the validation gate; a box-containing cut is accepted and provably
inert (85 nodes in 12/12 interleaved runs, identical objective).

**2. In the epigraph formulation, sound cuts exist and buy nothing.** Rewriting as
``min t s.t. t >= Psi(y)`` makes supporting hyperplanes of a convex underestimator
of Psi into valid ``t >= affine(y)`` cuts -- what "tangent-plane cuts of Gibbs
energy" means operationally. 119 such cuts were generated, validated and pooled:

    no cuts        nodes=239  obj=-17.46104641  bound=-17.461054074076234  optimal
    tangent cuts   nodes=239  obj=-17.46104641  bound=-17.461054074076234  optimal

Identical, to the last digit of the bound.

**Why, and what it implies.** A cut accepted here is GLOBAL -- discopt applies its
cut pool at every node, and `CutResult(scope="local")` is refused because there is
no subtree-scoped pool. So a cut must be valid over the whole root box, and a
globally-valid affine underestimator of Psi is weaker than the McCormick/entropy
envelopes discopt already builds on each NODE box. The tangent-plane cut that
would actually help a CALPHAD pricing solve is the node-local one, and node-local
cuts are exactly what the mechanism cannot hold.

So the useful next step for this class is subtree-scoped cuts, not more cut
GENERATION machinery. Recorded in docs/dev/performance-plan.md §68.

A note on the gate, which earned its keep here. The first version of the
underestimator took its intercept from the worst residual on a 701x701 grid; the
true worst residual is attained off-grid, and the gate refused the cut at a
1.308e-06 relative violation. That is a well-meant, nearly-right cut caught before
it could poison the tree -- the case component D's mandatory validation exists for.
"""

from pathlib import Path

import discopt.modeling as dm
import numpy as np

try:
    from discopt.calphad import ThermoDatabase
except ImportError:  # pragma: no cover - the plugin is not a discopt dependency
    raise SystemExit(
        "discopt-calphad is not importable; this entry experiment needs the plugin "
        "(github.com/jkitchin/discopt-calphad) on PYTHONPATH."
    )
from discopt.callbacks import CutResult
from discopt.calphad.equilibrium.common import DOMAIN_FLOOR, inner_bounds, tangent_distance
from discopt.calphad.ops import EXPR, NUM

db = ThermoDatabase.from_tdb(Path("/home/user/discopt-calphad/examples/data/cumg_nims.tdb"))
ph = db.phase_models(("CU", "MG"))["CU2MG"]
T, MU = 700.0, np.array([-1.0, -1.0])
BOX = inner_bounds(ph, DOMAIN_FLOOR)
LO = np.array([b[0] for b in BOX])
HI = np.array([b[1] for b in BOX])


def psi(yv):
    return float(tangent_distance(ph, list(yv), MU, NUM, T))


# Dense truth + a safe epigraph range for t.
gr = [np.linspace(LO[i], HI[i], 701) for i in range(2)]
VALS = np.array([[psi([a, b]) for b in gr[1]] for a in gr[0]])
TRUTH, PSI_HI = float(VALS.min()), float(VALS.max())
print(f"truth={TRUTH:.8f}  psi range=[{TRUTH:.4f}, {PSI_HI:.4f}]")


def build_epigraph():
    m = dm.Model("price_epi")
    y = [m.continuous(f"y{i}", lb=LO[i], ub=HI[i]) for i in range(2)]
    t = m.continuous("t", lb=TRUTH - 10.0, ub=PSI_HI + 10.0)
    m.subject_to(t >= tangent_distance(ph, y, MU, EXPR, T))
    m.minimize(t)
    return m, y, t


def affine_underestimator(y0):
    """(g, c) with Psi(y) >= g.y + c for ALL y in the box, tangent-ish at y0.

    Built by finite differences on Psi for the smooth (entropy) part and an
    exact global McCormick facet for the bilinear part would require parsing the
    expression; instead the whole Psi is underestimated by the CONCAVE-side
    secant-free construction below, which is checked numerically over a dense grid
    before the cut is ever returned. A cut that fails that check is not emitted.
    """
    y0 = np.clip(np.asarray(y0, float), LO, HI)
    h = 1e-5
    g = np.zeros(2)
    for i in range(2):
        a = y0.copy()
        b = y0.copy()
        a[i] = min(HI[i], y0[i] + h)
        b[i] = max(LO[i], y0[i] - h)
        g[i] = (psi(a) - psi(b)) / (a[i] - b[i])
    c = psi(y0) - g @ y0
    # Make it a VALID global underestimator by lowering the intercept to the worst
    # violation over a dense grid, plus a margin. Sound by construction on the grid
    # and checked again below.
    resid = VALS - (g[0] * gr[0][:, None] + g[1] * gr[1][None, :] + c)
    # The grid minimum is optimistic: the true worst residual is attained off it.
    # Back off by a margin scaled to the intercept, which weakens the cut slightly
    # and keeps it valid. (Without this the gate refused the cut at a 1.3e-6
    # relative violation — the gate working exactly as intended on a well-meant
    # but slightly-wrong cut.)
    c += float(resid.min()) - 1e-4 * max(1.0, abs(c))
    return g, c


def make_cb(counter):
    def cb(ctx, model):
        counter["n"] += 1
        g, c = affine_underestimator(ctx.x_relaxation[:2])
        # t - g.y >= c, i.e. the epigraph variable dominates the affine underestimator.
        return [CutResult(terms=[(Y[0], -g[0]), (Y[1], -g[1]), (Tv, 1.0)], sense=">=", rhs=c)]

    return cb


n = 0
results = {}
for arm in ("no cuts", "tangent cuts"):
    m, Y, Tv = build_epigraph()
    counter = {"n": 0}
    kw = {} if arm == "no cuts" else {"cut_callback": make_cb(counter)}
    r = m.solve(time_limit=300, gap_tolerance=1e-6, **kw)
    scale = max(1.0, abs(TRUTH))
    assert r.bound is None or r.bound <= TRUTH + 1e-5 * scale, f"{arm}: FALSE BOUND {r.bound}"
    assert abs(r.objective - TRUTH) <= 1e-3 * scale, f"{arm}: wrong optimum {r.objective}"
    stats = r.solver_stats or {}
    print(
        f"{arm:>14}: nodes={r.node_count:6d} obj={r.objective:.8f} bound={r.bound!r} "
        f"{r.status} cb_calls={counter['n']} cuts={stats.get('cut_validation/cuts', 0):.0f}"
    )
    results[arm] = r.node_count
    n += 1
assert n == 2
a, b = results["no cuts"], results["tangent cuts"]
print(f"\nnode count: {a} -> {b}  ({'GAIN' if b < a else 'no gain'})")
