"""OA must not re-linearize a row the master already carries exactly.

``_decompose_model`` splits every evaluator row in two: one whose coefficients
``_extract_body_coeffs`` recovers symbolically goes into ``linear_A_rows`` and
enters the master EXACTLY; everything else goes into ``nonlinear_indices``. An
affine row is convex, so the convexity mask marked it true and the cut
generators linearized it *as well* -- re-deriving, from ``g(x_bar)`` and
``J(x_bar)`` in floating point, a row the master already had.

That re-derivation is not exact, and the failure is a false infeasibility
certificate, not a rounding nuisance. Measured on ``bchoco07`` (2026-09-20)
before the fix:

* all 97 rows classified convex were affine rows already in the master;
* every cut's coefficient vector came back bit-identical (``max|dA| = 0.0``);
* constraint 95's right-hand side came back **2.747e-07** off the symbolic one.

``_add_oa_cuts`` appends a ``==`` row in both directions, so the master held
``a.x <= b`` together with ``a.x >= b + 2.7e-07`` -- infeasible by construction.
The master MILP reported infeasible at node 1 of iteration 0, with zero NLP
subproblems run, and ``solve_oa`` published ``status="infeasible"``: a claim
that the model has no solution at all (CLAUDE.md §1). SCIP could not prove
``bchoco07`` infeasible in 19570 nodes, and the minimum total violation of that
"infeasible" master, measured with an elastic LP, was 2.747e-07 -- carried
entirely by that one duplicated row.

The fix restricts every cut generator to ``oa_cut_mask`` (convex AND not already
exact in the master) while ``oa_constraint_mask`` stays the pure convexity
certificate the no-good-cut soundness gates read. It is bound-neutral in exact
arithmetic, because the skipped cut *is* the row already present: bchoco06/07/08
all kept ``bound=1.0`` to the digit with their cut counts at 0 (bchoco06 had
been accumulating 5740 of these duplicates without moving the bound).

Bounded by ITERATIONS, never by the wall clock: a budget of ``max_nodes`` is
deterministic and machine-independent, and a test whose arms are cut off by a
timer compares two different amounts of work (the lesson from the #1379 CI
failure). ``time_limit`` here is set far above what the budget needs so it can
never bind.
"""

import discopt.solvers.oa as oa
import numpy as np
import pytest
from discopt import Model
from discopt.modeling.core import from_nl

pytestmark = pytest.mark.smoke

_NL = "python/tests/data/minlplib_nl/bchoco07.nl"
_BUDGET = 1  # the defect fires at iteration 0; one iteration is enough
_TL = 300.0  # deliberately non-binding


def _affine_model():
    """A model whose rows are all affine -- every one already exact in the master."""
    m = Model("affine")
    x = m.continuous("x", lb=0.0, ub=10.0)
    y = m.binary("y")
    # Large coefficients are what make the re-derived right-hand side drift.
    m.subject_to(78000.0 * x + 1.0 * y == 78000.0)
    m.subject_to(x + y <= 11.0)
    m.minimize(x)
    return m


def test_affine_rows_are_excluded_from_cut_generation():
    """The cut mask must drop exactly the rows already exact in the master."""
    d = oa._decompose_model(_affine_model())
    assert d.oa_cut_mask is not None
    assert len(d.oa_cut_mask) == d.evaluator.n_constraints

    exact_rows = set(range(d.evaluator.n_constraints)) - set(d.nonlinear_indices)
    assert exact_rows, "probe is vacuous: this model has no exactly-represented row"
    for row in sorted(exact_rows):
        assert not d.oa_cut_mask[row], f"row {row} is already exact in the master but eligible"


def test_the_convexity_certificate_is_not_collateral_damage():
    """``oa_constraint_mask`` must stay the pure convexity verdict.

    ``_assignment_proven_infeasible`` admits a no-good cut only when
    ``all(constraint_convex_mask)`` holds. Folding "already exact in the master"
    into that mask would silently disable the proven-infeasible path on every
    model with an affine row, so the two masks are deliberately separate lists.
    """
    d = oa._decompose_model(_affine_model())
    assert d.oa_constraint_mask is not None
    # Every row here is affine, hence convex.
    assert all(d.oa_constraint_mask), d.oa_constraint_mask
    assert d.oa_cut_mask != d.oa_constraint_mask


def test_a_convex_nonlinear_row_is_still_linearized():
    """The fix must not switch OA off: a genuinely convex nonlinear row keeps its cut."""
    m = Model("convex_nl")
    x = m.continuous("x", lb=-5.0, ub=5.0)
    y = m.binary("y")
    m.subject_to(x * x - 4.0 <= 0.0)  # convex, nonlinear -> needs a cut
    m.subject_to(x + y <= 3.0)  # affine -> already exact
    m.minimize(x)

    d = oa._decompose_model(m)
    assert d.oa_cut_mask is not None
    eligible = [i for i, ok in enumerate(d.oa_cut_mask) if ok]
    assert eligible, "no row eligible for an OA cut -- the fix disabled OA entirely"
    for row in eligible:
        assert row in set(d.nonlinear_indices)
        assert d.oa_constraint_mask[row]


def test_oa_does_not_declare_bchoco07_infeasible():
    """The regression: a duplicated affine row must not certify the model empty.

    Asserted as an invariant, not a pinned float. "Infeasible" is a claim that no
    feasible point exists anywhere; the search reaching a limit is not evidence
    for it, so any limit/unknown exit is acceptable here and ``infeasible`` is
    not.
    """
    m = from_nl(_NL)
    r = m.solve(time_limit=_TL, max_nodes=_BUDGET, solver="mip-nlp", mip_nlp_method="oa")

    assert r.status != "infeasible", (
        f"OA certified bchoco07 infeasible (status={r.status!r}); the master was "
        "made empty by a round-off-perturbed duplicate of a row it already held"
    )
    # An exit with no incumbent must not smuggle the claim in through the other
    # certificate fields either.
    assert r.objective is None or np.isfinite(r.objective)
    assert not (r.x is None and r.gap_certified)


def test_no_cut_duplicates_a_row_the_master_already_carries():
    """General invariant on the real instance, independent of the status it exits with.

    This is the property the certificate rests on: a cut may tighten the master,
    but it must never be a second, numerically different copy of a row already
    in it. Before the fix bchoco07 produced 155 such rows.
    """
    m = from_nl(_NL)
    d = oa._decompose_model(m)

    exact_rows = set(range(d.evaluator.n_constraints)) - set(d.nonlinear_indices)
    assert exact_rows, "probe is vacuous: bchoco07 has no exactly-represented row"

    assert d.oa_cut_mask is not None
    duplicated = sorted(row for row in exact_rows if d.oa_cut_mask[row])
    assert not duplicated, (
        f"{len(duplicated)} rows are already exact in the master yet eligible for "
        f"an OA cut: {duplicated[:10]}"
    )
