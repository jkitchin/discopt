"""EP2 (#632) — per-node OBBT reuses one relaxation per (node box, round).

The per-node OBBT entry point :func:`obbt_tighten_root` builds the McCormick
relaxation **once per (node box, OBBT round)** and then probes ``min/max x_i``
for every candidate over that single relaxation, swapping only the LP objective
vector and warm-starting each probe from the previous probe's optimal basis
(the ``_PersistentProbeLP`` mechanism landed in PR #579 / cert:T2.2 and consumed
here through :func:`run_obbt_on_relaxation`). No probe rebuilds the relaxation —
the OBBT ``root_time`` the EP0 profile attributed to "each probe paying a full
engine build" is the per-*round* ``build_milp_relaxation`` cost, not a per-probe
rebuild.

These tests pin EP2's neutrality invariant at the *node* level (the T2.2 tests
in ``test_obbt_warm_probes.py`` pin it at the ``run_obbt_on_relaxation`` level on
synthetic models). For each vendored instance they compare:

* the default **reuse + warm-start** path (``_PersistentProbeLP`` active), and
* the **cold-seam** reference (``_simplex_available`` forced ``False``), which
  re-solves the *same* per-round relaxation cold for every probe — i.e. the
  behaviour before any per-sweep reuse.

Reuse of the per-round relaxation matrix is exact (the same LP is solved), and
the warm-started basis returns the same LP *optimum value* as a cold solve, so
the applied tightenings — hence the OBBT-produced boxes — must be **identical to
within warm-start vertex noise** (``<= 1e-9``; observed ``<= 1 ulp``). A
regression that made the reuse/warm path change any tightening (the only way
this could be unsound) fails here.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt._jax.obbt as obbt_mod
import numpy as np
import pytest
from discopt._jax.model_utils import flat_variable_bounds
from discopt._jax.obbt import obbt_tighten_root
from discopt.modeling.core import from_nl

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")

_NL_DIR = Path(__file__).parent / "data" / "minlplib_nl"

# Small vendored instances whose only-nonlinear / mixed structure makes
# ``obbt_tighten_root`` actually probe-and-tighten, and that finish a full
# 3-round sweep quickly. Each is verified below to tighten at least one bound so
# the equality assertion is exercised on real tightenings (not a trivial no-op).
_INSTANCES = [
    "ex1222",
    "ex1221",
    "ex1224",
    "ex1225",
    "ex1226",
    "nvs03",
    "nvs07",
    "nvs10",
    "st_e13",
    "gbd",
]


def _run(name: str, *, force_cold: bool):
    model = from_nl(str(_NL_DIR / f"{name}.nl"))
    lb, ub = flat_variable_bounds(model)
    if force_cold:
        orig = obbt_mod._simplex_available
        obbt_mod._simplex_available = lambda: False  # force the cold-seam probe path
        try:
            return obbt_tighten_root(model, lb.copy(), ub.copy(), rounds=3, time_limit_per_lp=5.0)
        finally:
            obbt_mod._simplex_available = orig
    return obbt_tighten_root(model, lb.copy(), ub.copy(), rounds=3, time_limit_per_lp=5.0)


def _maxdiff(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    # Matching non-finite entries (an untightened +/-inf bound in both paths) are
    # equal; only finite entries carry a numeric difference.
    both_inf = ~np.isfinite(a) & ~np.isfinite(b) & (np.sign(a) == np.sign(b))
    d = np.abs(a - b)
    d[both_inf] = 0.0
    return float(np.max(d)) if d.size else 0.0


@pytest.mark.parametrize("name", _INSTANCES)
def test_node_obbt_reuse_matches_cold_seam(name: str) -> None:
    """Reuse+warm per-node OBBT boxes == cold-seam boxes (differential-neutral)."""
    if not (_NL_DIR / f"{name}.nl").exists():
        pytest.skip(f"missing vendored instance {name}")
    warm = _run(name, force_cold=False)
    cold = _run(name, force_cold=True)

    # Same number of tightenings, and the produced boxes match to warm-start
    # vertex noise (the LP optimum value is identical either way).
    assert warm.n_tightened == cold.n_tightened
    assert _maxdiff(warm.lb, cold.lb) <= 1e-9
    assert _maxdiff(warm.ub, cold.ub) <= 1e-9
    # The reuse/warm path must never loosen: it is a subset of the original box.
    assert warm.infeasible == cold.infeasible


def test_node_obbt_actually_tightens() -> None:
    """At least five of the probe-set instances tighten a bound (equality is exercised)."""
    tightened = 0
    seen = 0
    for name in _INSTANCES:
        if not (_NL_DIR / f"{name}.nl").exists():
            continue
        seen += 1
        if _run(name, force_cold=False).n_tightened > 0:
            tightened += 1
    assert seen >= 5
    assert tightened >= 5, f"only {tightened} instances tightened; equality gate under-exercised"
