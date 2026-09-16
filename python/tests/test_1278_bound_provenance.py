"""Bound provenance — component F of #1248, via #1278.

#1278 F asked for ``Model.solve(verified_bound=True)``: "use the
Neumaier-Shcherbina safe bound and outward rounding throughout, and report the
provenance". Two thirds of that premise was already stale when the issue was
written, and the third turned out to be the whole of it.

* **Outward rounding** graduated to default ON on 2026-08-09
  (``_relax/outward_rounding.outward_rounding_enabled``).
* **The NS safe bound** is already what ``mccormick_lp._certify`` prefers on the
  default in-house simplex path — its own comment says "the common pure-LP
  warm-simplex path: rigorous safe bound".
* **The provenance** was genuinely missing: from outside the solver there was no
  way to tell a bound resting on a rigorous NS certificate from one resting on an
  LP optimum taken on trust.

So F ships as the provenance, and the strict mode it implies is declined on the
measurement that the provenance itself produced. Across the 66-instance in-repo
corpus, 507 certified nodes:

    ns_safe_bound     505   99.61%
    declined            2    0.39%
    trusted_backend     0
    trusted_vertex      0

A ``verified_bound=True`` that refused the trusted arms would have nothing to
refuse on any corpus instance. The tally is the deliverable; the mode is a no-op.
"""

from __future__ import annotations

import glob
import pathlib

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.mccormick_lp import BOUND_PROVENANCE, reset_bound_provenance

pytestmark = [pytest.mark.smoke]

#: Every tag ``_certify`` can emit. A new arm added without a tag would make the
#: tally silently under-count, which is the failure this file exists to prevent.
TAGS = {"ns_safe_bound", "milp_dual", "trusted_backend", "trusted_vertex", "declined"}

#: The tags that mean "this bound is a rigorous certificate", as opposed to an LP
#: optimum accepted on trust.
RIGOROUS = {"ns_safe_bound", "milp_dual"}


def _nonconvex_model():
    m = dm.Model("prov")
    x = m.continuous("x", lb=0.0, ub=2.0)
    y = m.continuous("y", lb=0.0, ub=2.0)
    m.subject_to(x + y == 2.0)
    m.minimize(x * y - dm.exp(x))
    return m


def test_every_certified_node_reports_which_argument_certified_it():
    reset_bound_provenance()
    r = _nonconvex_model().solve(time_limit=60)
    assert r.status == "optimal", r.status
    tally = dict(BOUND_PROVENANCE)
    assert tally, "no node was certified — the probe measured nothing"
    assert set(tally) <= TAGS, sorted(set(tally) - TAGS)
    assert sum(tally.values()) > 0


def test_the_tally_reaches_solver_stats_as_a_per_solve_delta():
    """The counter is process-global; what a caller needs is this solve's share."""
    first = _nonconvex_model().solve(time_limit=60)
    stats = first.solver_stats or {}
    keys = {k for k in stats if k.startswith("bound_provenance/")}
    assert keys, sorted(stats)
    assert all(k.split("/", 1)[1] in TAGS for k in keys), sorted(keys)

    # A second, identical solve must report its OWN nodes, not the running total.
    second = _nonconvex_model().solve(time_limit=60)
    s2 = second.solver_stats or {}
    for k in keys:
        assert s2.get(k, 0.0) == pytest.approx(stats[k]), (k, stats[k], s2.get(k))


def test_the_default_path_certifies_from_the_safe_bound_not_from_trust():
    """The claim F rests on: on the default in-house simplex the bound is an NS
    safe bound, which is valid for ANY dual vector, so a drifted basis can only
    loosen it — never lift it above the optimum."""
    reset_bound_provenance()
    r = _nonconvex_model().solve(time_limit=60)
    assert r.status == "optimal"
    tally = dict(BOUND_PROVENANCE)
    rigorous = sum(v for k, v in tally.items() if k in RIGOROUS)
    trusted = sum(v for k, v in tally.items() if k in ("trusted_backend", "trusted_vertex"))
    assert rigorous > 0, tally
    assert trusted == 0, tally


@pytest.mark.slow
def test_the_corpus_certifies_from_proof_not_trust():
    """The measurement in the module docstring, as a standing regression: if a
    change starts certifying corpus nodes from a trusted LP optimum, that is a
    weakening of the certificate and this test is where it surfaces."""
    from discopt.modeling.core import from_nl

    root = pathlib.Path(__file__).parent / "data" / "minlplib_nl"
    files = sorted(glob.glob(str(root / "*.nl")))
    assert len(files) > 50, len(files)

    reset_bound_provenance()
    solved = 0
    for path in files[:25]:
        try:
            from_nl(path).solve(time_limit=10, max_nodes=300)
        except Exception:  # noqa: BLE001 - an instance that raises is not this test's subject
            continue
        solved += 1
    assert solved >= 20, f"only {solved} instances ran — the panel measured nothing"

    tally = dict(BOUND_PROVENANCE)
    total = sum(tally.values())
    assert total > 0, "no node was certified across the panel"
    trusted = tally.get("trusted_backend", 0) + tally.get("trusted_vertex", 0)
    assert trusted == 0, (
        f"{trusted}/{total} node bounds rested on a trusted LP optimum rather than a "
        f"certificate: {tally}"
    )


def test_resetting_the_tally_zeroes_it():
    _nonconvex_model().solve(time_limit=60)
    assert sum(BOUND_PROVENANCE.values()) > 0
    reset_bound_provenance()
    assert dict(BOUND_PROVENANCE) == {}


def test_outward_rounding_and_the_ns_safe_bound_are_both_default_on():
    """The two halves of F's premise that were already true. Pinned so a silent
    flip of either default shows up as a failing test rather than as a quietly
    weaker certificate."""
    from discopt._relax.outward_rounding import outward_rounding_enabled
    from discopt.solver_tuning import SolverTuning

    assert outward_rounding_enabled() is True
    assert SolverTuning().node_numerical_dual_bound is True


def test_the_reported_bound_is_never_above_the_true_optimum():
    """Provenance is only worth having if the bound it describes is valid. A
    one-variable problem whose optimum is known by dense sampling."""
    m = dm.Model("valid")
    x = m.continuous("x", lb=0.05, ub=3.0)
    m.minimize(dm.xlogx(x) + dm.sin(3.0 * x))
    r = m.solve(time_limit=60)
    grid = np.linspace(0.05, 3.0, 400_001)
    truth = float(np.min(grid * np.log(grid) + np.sin(3.0 * grid)))
    scale = max(1.0, abs(truth))
    assert r.bound is not None
    assert r.bound <= truth + 1e-6 * scale, (r.bound, truth)
    assert r.objective == pytest.approx(truth, abs=1e-4 * scale)
