"""#844 follow-up: a child whose relaxation cannot be resolved must not be dropped.

``child()`` used to silently discard any node whose LP came back without a bound.
But ``None`` conflated two very different outcomes:

* the LP feasible set over the child's box is provably empty — a rigorous fathom,
  since the McCormick polytope is a valid OUTER approximation, so an empty
  relaxation means the subtree holds no feasible point;
* the LP solve simply **failed** (numerical error, time limit, or an ``infeasible``
  claim with no Farkas proof) — in which case the subtree is *not* ruled out.

Dropping the second kind removes live space from the search. If the heap then
exhausts, the engine declares ``status="optimal"`` over a region it never examined:
a false optimality certificate, the worst error class (CLAUDE.md §1).

The fix threads a verdict out of ``node_relax`` (``"optimal"`` / ``"fathom"`` /
``"unresolved"``, where ``"fathom"`` requires a *verified Farkas dual ray*) and
folds an unresolved child's parent bound — a valid lower bound over the child's box
— into ``unresolved_lb``, the floor that already gates the optimality claim.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt._jax.lp_spatial_bb as lpsb  # noqa: E402
import discopt.modeling as dm  # noqa: E402
import pytest  # noqa: E402


def _model() -> dm.Model:
    """Pure-integer MINIMIZE whose incremental structure does NOT build, so the
    engine relaxes through ``_relax_bound`` — the function this test injects into.

    Uses a TRILINEAR coupling. The original fixture used bilinear *constraints*,
    which declined the fast path only because a constraint on the product lifted to
    a 5th row over the term's own columns and the structure miscounted it as an
    envelope row. #861 fixed that, the model started building a structure, the
    engine stopped calling ``_relax_bound`` — and the injection below silently
    stopped firing. The test's own vacuity guard caught it (that is what the guard
    is for), but the fixture had to become durable: trilinear lifts are outside the
    closed-form patch's covered families by design, so this decline is a property of
    the mathematics rather than of a bug. The precondition is asserted below.
    """
    m = dm.Model("unresolved_child")
    xs = [m.integer(f"x{i}", lb=0, ub=8) for i in range(4)]
    m.minimize(sum((i + 1) * xs[i] for i in range(4)))
    for i in range(2):
        m.subject_to(xs[i] * xs[i + 1] * xs[i + 2] >= 6)
    return m


def _incremental_model() -> dm.Model:
    """Same shape, but its structure DOES build — so the engine relaxes through the
    incremental fast path instead. Since #861 that is the common case, and the
    unresolved-child soundness property must hold there too."""
    m = dm.Model("unresolved_child_fast")
    xs = [m.integer(f"x{i}", lb=0, ub=8) for i in range(4)]
    m.minimize(sum(xs[i] * xs[i + 1] for i in range(3)))
    for i in range(3):
        m.subject_to(xs[i] + xs[i + 1] >= 5)
    return m


def _structure_builds(model) -> bool:
    from discopt._jax.incremental_mccormick import IncrementalMcCormickLP
    from discopt._jax.term_classifier import classify_nonlinear_terms

    return bool(IncrementalMcCormickLP(model, classify_nonlinear_terms(model), deadline=None).ok)


def test_fixtures_exercise_the_paths_they_claim():
    """Guard the guards: the injection points below are path-specific, so a fixture
    that silently switched paths would make them vacuous (exactly what #861 did)."""
    assert not _structure_builds(_model()), "cold-path fixture now builds a structure"
    assert _structure_builds(_incremental_model()), "fast-path fixture no longer builds"


def test_baseline_is_certifiable():
    """Control: undisturbed, this instance certifies. Without this the injection
    test below could pass vacuously (never having had a certificate to lose)."""
    res = lpsb.solve_lp_spatial_bb(_model(), time_limit=30.0, gap_tolerance=1e-4)
    assert res is not None
    assert res.status == "optimal", f"baseline did not certify: {res.status}"


def test_failed_child_relaxation_never_yields_a_false_certificate(monkeypatch):
    """Inject unresolvable child LPs into a PROVABLY FEASIBLE model and require the
    engine to make no certified claim about the space it could not examine.

    Measured before the fix, with 3 injected relaxation failures::

        without fix:  status="infeasible"     <-- FALSE INFEASIBILITY CERTIFICATE
        with fix:     status="time_limit"     <-- honest "could not resolve"

    The model is demonstrably feasible (the undisturbed solve certifies optimum
    24.0), so ``infeasible`` was a false certificate: every failed child was dropped
    silently, the heap emptied with no incumbent, and the engine concluded the
    feasible set was empty. Folding an unresolved child into ``unresolved_lb`` makes
    that conclusion impossible.
    """
    truth = lpsb.solve_lp_spatial_bb(_model(), time_limit=30.0, gap_tolerance=1e-4)
    assert truth is not None and truth.status == "optimal", "control lost its certificate"

    real = lpsb._relax_bound
    state = {"calls": 0, "failures": 0}

    def flaky(model, terms, lb, ub, **kw):
        # ``**kw`` passes through whatever optional arguments the real
        # ``_relax_bound`` grows (``deadline`` since #860) so the injection keeps
        # testing the unresolved-child path rather than failing on a signature.
        state["calls"] += 1
        if state["calls"] > 3:  # let the root and first nodes succeed
            state["failures"] += 1
            return None
        return real(model, terms, lb, ub, **kw)

    monkeypatch.setattr(lpsb, "_relax_bound", flaky)
    res = lpsb.solve_lp_spatial_bb(_model(), time_limit=30.0, gap_tolerance=1e-4)

    # The injection must actually have fired, or this asserts nothing.
    assert state["failures"] > 0, "injection never failed a solve — test would be vacuous"
    assert res is not None

    # THE soundness property: no certified verdict over unexamined space.
    assert res.status != "infeasible", (
        "FALSE INFEASIBILITY: engine declared a feasible model infeasible after "
        f"dropping {state['failures']} unresolvable children (true optimum "
        f"{truth.objective})"
    )
    if res.status == "optimal":
        assert res.objective == pytest.approx(truth.objective, rel=1e-6), (
            f"FALSE OPTIMUM: certified {res.objective} over unexamined space "
            f"(true optimum {truth.objective})"
        )


def test_failed_child_on_the_incremental_path_never_yields_a_false_certificate(monkeypatch):
    """The same soundness property, on the FAST path.

    The test above injects into ``_relax_bound``, which the engine only calls when
    the incremental structure declined. Since #861 most in-scope models build one,
    so that injection no longer covers the common case: the engine relaxes through
    ``IncrementalMcCormickLP.solve``. Inject there instead and require the same
    property — no certified verdict over space the engine could not examine.
    """
    from discopt._jax.incremental_mccormick import IncrementalMcCormickLP

    truth = lpsb.solve_lp_spatial_bb(_incremental_model(), time_limit=30.0, gap_tolerance=1e-4)
    assert truth is not None and truth.status == "optimal", "control lost its certificate"

    real_solve = IncrementalMcCormickLP.solve
    state = {"calls": 0, "failures": 0}

    def flaky(self, lb, ub, **kw):
        state["calls"] += 1
        if state["calls"] > 3:  # let the root and first nodes succeed
            state["failures"] += 1
            return None, None, None  # unresolved: no bound, no point, no basis
        return real_solve(self, lb, ub, **kw)

    monkeypatch.setattr(IncrementalMcCormickLP, "solve", flaky)
    res = lpsb.solve_lp_spatial_bb(_incremental_model(), time_limit=30.0, gap_tolerance=1e-4)

    assert state["failures"] > 0, "injection never failed a node solve — test would be vacuous"
    assert res is not None
    assert res.status != "infeasible", (
        "FALSE INFEASIBILITY on the incremental path: engine declared a feasible model "
        f"infeasible after {state['failures']} unresolvable children (true optimum "
        f"{truth.objective})"
    )
    if res.status == "optimal":
        assert res.objective == pytest.approx(truth.objective, rel=1e-6), (
            f"FALSE OPTIMUM on the incremental path: certified {res.objective} over "
            f"unexamined space (true optimum {truth.objective})"
        )


def test_verdict_contract_is_tri_state():
    """``node_relax`` must expose a verdict, and 'fathom' must never be the verdict
    for an uncertified failure — that is the distinction the fix rests on."""
    import inspect

    src = inspect.getsource(lpsb.solve_lp_spatial_bb)
    assert '"fathom"' in src and '"unresolved"' in src, "verdict contract missing"
    # a fathom must require the Farkas proof, not merely an 'infeasible' label
    assert '_st == "infeasible" and _farkas' in src, (
        "fathoming must require a verified Farkas dual ray"
    )
