"""Would a node-local cut even be VALID on a real CALPHAD pricing instance?

Settles a question I raised without evidence. §68's first draft, and a comment on
#1248, concluded that the next useful step for this class was subtree-scoped cuts,
reasoning that a globally-valid cut is too weak so a node-local one would help.
That is inference, not evidence, and CLAUDE.md §4 forbids shipping it as a
recommendation. This measures it.

The cut a CALPHAD plugin would actually write is a tangent plane of the Gibbs
energy. A tangent plane underestimates Psi on a box only if Psi is CONVEX on that
box, so the question is how often Psi is certifiably convex on a sub-box of the
real CU2MG phase. The verdict used is the ENGINE's own (`_try_convex_lift`, via
the composite_convex coverage tag), not a separate convexity opinion, so this is
what the solver would conclude.

RESULT -- never, at any depth:

                       w=1     w=0.5   w=0.25  w=0.1   w=0.02  w=0.005 w=0.001
        CU2MG  Psi     0/12    0/12    0/12    0/12    0/12    0/12    0/12
    CONTROL convex    12/12   12/12   12/12   12/12   12/12   12/12   12/12

Down to a 1000x subdivision of the root box. Structural, not a budget artefact:
the phase's bilinear cross terms have Hessian [[0,c],[c,0]], eigenvalues +-|c| on
every box, so interval-Gershgorin can never be sign-definite. The convex control
fires at every width, which is what makes the zeros mean something (CLAUDE.md §6).

So a tangent-plane cut is unsound at every node at every depth, and subtree
scoping would not let a plugin write the cut it wants. The sound alternative -- an
alphaBB-corrected tangent -- is machinery discopt already runs in its relaxation.
The subtree-scoped-cuts recommendation is withdrawn; no follow-up work is implied.

Requires discopt-calphad on the path; discopt does not depend on it.
"""

from pathlib import Path

import discopt.modeling as dm
import numpy as np
from discopt._relax.uniform_relax import build_uniform_relaxation

try:
    from discopt.calphad import ThermoDatabase
except ImportError:  # pragma: no cover - the plugin is not a discopt dependency
    raise SystemExit(
        "discopt-calphad is not importable; this entry experiment needs the plugin "
        "(github.com/jkitchin/discopt-calphad) on PYTHONPATH."
    )
from discopt.calphad.equilibrium.common import DOMAIN_FLOOR, inner_bounds, tangent_distance
from discopt.calphad.ops import EXPR

db = ThermoDatabase.from_tdb(Path("/home/user/discopt-calphad/examples/data/cumg_nims.tdb"))
ph = db.phase_models(("CU", "MG"))["CU2MG"]
T, MU = 700.0, np.array([-1.0, -1.0])
BOX = inner_bounds(ph, DOMAIN_FLOOR)
LO = np.array([b[0] for b in BOX])
HI = np.array([b[1] for b in BOX])


def certified_convex(lo, hi):
    m = dm.Model("sub")
    y = [m.continuous(f"y{i}", lb=float(lo[i]), ub=float(hi[i])) for i in range(2)]
    m.minimize(tangent_distance(ph, y, MU, EXPR, T))
    rel = build_uniform_relaxation(m)
    return "composite_convex" in {k for k, _ in rel.coverage.values()}


# A convex control, so a row of zeros means something (CLAUDE.md §6).
def control_convex(lo, hi):
    m = dm.Model("ctl")
    y = [m.continuous(f"y{i}", lb=float(lo[i]), ub=float(hi[i])) for i in range(2)]
    m.minimize(dm.exp(y[0] + y[1]) + dm.exp(y[0] - y[1]))
    rel = build_uniform_relaxation(m)
    return "composite_convex" in {k for k, _ in rel.coverage.values()}


rng = np.random.default_rng(11)
WIDTHS = [1.0, 0.5, 0.25, 0.1, 0.02, 0.005, 0.001]
SAMPLES = 12
print(f"{'':>22} " + "  ".join(f"w={w:<6g}" for w in WIDTHS))
for label, fn in (("CU2MG  Psi", certified_convex), ("CONTROL convex", control_convex)):
    cells, tot = [], 0
    for w in WIDTHS:
        hits = 0
        for _ in range(SAMPLES):
            span = (HI - LO) * w
            lo = LO + rng.random(2) * ((HI - LO) - span)
            hits += bool(fn(lo, lo + span))
        cells.append(f"{hits:2d}/{SAMPLES}  ")
        tot += hits
    print(f"{label:>22} " + "".join(cells))
    if label.startswith("CONTROL"):
        assert tot == len(WIDTHS) * SAMPLES, f"the control certified only {tot} times"
print(f"\nEXECUTED_PROBES={2 * len(WIDTHS) * SAMPLES}")
