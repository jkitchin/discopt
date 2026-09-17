"""Would a node-local cut even be VALID on a real CALPHAD pricing instance?

Settles a question raised without evidence. An earlier draft of
``docs/dev/performance-plan.md`` §68, and a comment on #1248, concluded that the
next useful step for this class was subtree-scoped cuts, reasoning that a
globally-valid cut is too weak so a node-local one would help. That is inference,
not evidence -- "the global cut is too weak" does not establish that a local one
would be strong enough -- and shipping it as a recommendation is the speculative
hand-off CLAUDE.md §4 forbids. So it is measured.

The cut a CALPHAD plugin would actually write is a **tangent plane of the Gibbs
energy**, which underestimates Psi on a box only where Psi is **convex**. The
question is therefore how often Psi is certifiably convex on a sub-box. The
verdict used is the ENGINE's own (``_try_convex_lift``, via the
``composite_convex`` coverage tag), so this is what the solver would conclude, not
a separate convexity opinion.

Run over **every mixing phase** the shipped assessments provide, not just CU2MG,
because a verdict about one named instance is a gate probe rather than a result
(CLAUDE.md §2). A convex control runs at every width so a row of zeros means
"abstained", not "the probe is broken" (CLAUDE.md §6).

RESULT -- 0 certifiably-convex sub-boxes across 6 phases and 588 probes, at every
width down to a 1000x subdivision of the root box, while the control fires 12/12
everywhere.

Two DISTINCT reasons sit behind those zeros, and they should not be lumped:

* The five ``n_y == 1`` phases (LIQUID, FCC_A1, HCP_A3) are not eligible for the
  composite lift at all -- it requires at least two variables. For them the
  relevant fact is the other way round: the engine already emits the EXACT 1-D
  secant/tangent envelope per node box (that is #1277's fix), so a user-supplied
  tangent line has nothing to add.
* The one genuine multivariate phase, ``CU2MG`` (n_y = 2), is eligible and never
  certifies, at any depth. That is structural rather than a budget artefact: a CEF
  phase's bilinear cross terms have Hessian ``[[0, c], [c, 0]]`` with eigenvalues
  ``+-|c|`` on every box, so interval-Gershgorin can never be sign-definite.

Either way no user tangent-plane cut has anything to contribute: where the cut
would be sound the engine already emits the exact envelope, and where the engine
abstains the cut is not sound. **Subtree scoping would not change either half.**
The sound alternative -- an alphaBB-corrected tangent -- is machinery discopt
already runs in its own relaxation. The subtree-scoped-cuts recommendation is
withdrawn; no follow-up work is implied.

Requires discopt-calphad on the path; discopt does not depend on it.
"""

from __future__ import annotations

from pathlib import Path

import discopt.modeling as dm
import numpy as np
from discopt._relax.uniform_relax import build_uniform_relaxation

try:
    from discopt.calphad import ThermoDatabase
    from discopt.calphad.equilibrium.common import DOMAIN_FLOOR, inner_bounds, tangent_distance
    from discopt.calphad.ops import EXPR
except ImportError:  # pragma: no cover - the plugin is not a discopt dependency
    raise SystemExit(
        "discopt-calphad is not importable; this entry experiment needs the plugin "
        "(github.com/jkitchin/discopt-calphad) on PYTHONPATH."
    ) from None

DATA = Path("/home/user/discopt-calphad/examples/data")
T = 700.0
WIDTHS = [1.0, 0.5, 0.25, 0.1, 0.02, 0.005, 0.001]
SAMPLES = 12


def _mixing_phases():
    """Every phase with at least one site fraction, across both assessments."""
    out = []
    for tdb, els in ((DATA / "cumg_nims.tdb", ("CU", "MG")), (DATA / "alzn_mey.tdb", ("AL", "ZN"))):
        db = ThermoDatabase.from_tdb(tdb)
        for name, phase in db.phase_models(els).items():
            if phase.n_y > 0:
                out.append((f"{tdb.stem}:{name}", phase, np.full(len(els), -1.0)))
    assert out, "no mixing phase found in the shipped assessments"
    return out


def _psi_is_certifiably_convex(phase, mu, lo, hi) -> bool:
    m = dm.Model("sub")
    y = [m.continuous(f"y{i}", lb=float(lo[i]), ub=float(hi[i])) for i in range(len(lo))]
    m.minimize(tangent_distance(phase, y, mu, EXPR, T))
    rel = build_uniform_relaxation(m)
    return "composite_convex" in {k for k, _ in rel.coverage.values()}


def _control_is_certifiably_convex(lo, hi) -> bool:
    """A plainly convex two-variable objective: if THIS ever fails to certify, a
    row of zeros above says nothing about the phases."""
    m = dm.Model("ctl")
    y = [m.continuous(f"y{i}", lb=float(lo[i]), ub=float(hi[i])) for i in range(2)]
    m.minimize(dm.exp(y[0] + y[1]) + dm.exp(y[0] - y[1]))
    rel = build_uniform_relaxation(m)
    return "composite_convex" in {k for k, _ in rel.coverage.values()}


def main() -> int:
    rng = np.random.default_rng(11)
    probes = 0
    phase_hits = 0

    print(f"{'':>26} " + "  ".join(f"w={w:<6g}" for w in WIDTHS))
    for label, phase, mu in _mixing_phases():
        box = inner_bounds(phase, DOMAIN_FLOOR)
        lo0 = np.array([b[0] for b in box])
        hi0 = np.array([b[1] for b in box])
        cells = []
        for w in WIDTHS:
            hits = 0
            for _ in range(SAMPLES):
                span = (hi0 - lo0) * w
                lo = lo0 + rng.random(len(lo0)) * ((hi0 - lo0) - span)
                hits += bool(_psi_is_certifiably_convex(phase, mu, lo, lo + span))
                probes += 1
            cells.append(f"{hits:2d}/{SAMPLES}  ")
            phase_hits += hits
        print(f"{label:>26} " + "".join(cells), flush=True)

    control_hits = 0
    cells = []
    for w in WIDTHS:
        hits = 0
        for _ in range(SAMPLES):
            lo = rng.random(2) * (1.0 - w)
            hits += bool(_control_is_certifiably_convex(lo, lo + w))
            probes += 1
        cells.append(f"{hits:2d}/{SAMPLES}  ")
        control_hits += hits
    print(f"{'CONTROL convex':>26} " + "".join(cells), flush=True)

    assert control_hits == len(WIDTHS) * SAMPLES, (
        f"the convex control certified only {control_hits}/{len(WIDTHS) * SAMPLES} times — "
        "the probe cannot say yes, so the phase rows mean nothing"
    )
    print()
    print(f"phases certifiably convex on any sampled sub-box: {phase_hits}")
    print(
        "VERDICT: "
        + (
            "a tangent-plane cut is never sound here — subtree scoping would not help"
            if phase_hits == 0
            else "some sub-box IS convex; a local tangent-plane cut could be sound there"
        )
    )
    print(f"EXECUTED_PROBES={probes}")
    return 0 if probes else 1


if __name__ == "__main__":
    raise SystemExit(main())
