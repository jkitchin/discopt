"""A ``Constraint`` with a non-zero ``rhs`` must be refused, never silently solved.

The defect (plan §6, 2026-07-30, filed by Card 4c Task 2). :class:`Constraint`
documents ``rhs`` as *always 0.0 in normalized form*, and the comparison operators
that build every row in the supported DSL fold the offset into ``body``. But
nothing enforced that, and the two halves of the stack disagree about what an
unnormalized row means:

* **26 modules read ``Constraint.body`` and never read ``.rhs``** — the whole
  relaxation stack (``dag_compiler``, ``relaxation_compiler``, ``milp_relaxation``,
  ``mccormick_subgradient``, ``term_classifier``, ``nonlinear_bound_tightening``,
  ``dependent_vars``, ``implied_integer``, the convexity certificate, ``bilevel/kkt``,
  Benders, RO). ``dag_compiler.compile_constraint`` is the seam: it compiles
  ``constraint.body`` and drops ``rhs`` on the floor.
* ``validation/feasibility`` (``signed = body - rhs``), the ``.nl`` and GAMS
  exporters, ``problem_classifier``, ``_relax/obbt`` and the Rust ``ConstraintRepr``
  (114 references in the presolve crate) all **do** honour it.

  *Correction (CLAUDE.md §11).* Issue #909 and the first write-up of this file
  both stated that "the exporters" honour ``rhs``, naming LP and MPS among them.
  **Measured on this tree, that is false for LP and MPS** — both recover the row's
  right-hand side from the *body* alone (``lp.py`` ``rows.append((..., -const))``,
  ``mps.py`` likewise), so a non-zero ``rhs`` is silently dropped:
  ``Constraint(w, ">=", 5.0)`` exported as ``c0: w >= 0`` in LP and as an empty
  ``RHS`` section in MPS. Only ``.nl`` (r-section ``2 5.0``) and GAMS
  (``c1.. w =g= 5.0;``) are faithful. The scoping below follows the measurement,
  not the write-up: ``.nl`` and GAMS pass ``for_solve=False`` and export such a
  row; LP and MPS keep the refusal, which converts a silently wrong export into a
  loud error.

So the row was *solved* as ``body sense 0`` and *verified* as ``body - rhs sense
0``: ``Constraint(w, ">=", 5.0)`` solved to ``w = 0`` while the equivalent
``w >= 5.0`` solved to ``w = 5``, and the verifier called the first one feasible.
A silent wrong answer, and — contrary to the original write-up, which believed only
the private ``_constraints.append`` path was affected — reachable straight through
the **public API**: ``Constraint`` is exported in ``discopt.modeling.__all__`` and
``m.subject_to(Constraint(w, ">=", 5.0))`` reproduced it exactly.

Direction of the fix (CLAUDE.md §3). The model boundary **refuses loudly**; the
solve path is not taught to honour ``rhs``. Threading ``rhs`` through would mean
correcting all 26 modules, each of which encodes the ``body sense 0`` form
structurally rather than arithmetically, and a partial job is strictly worse than
the status quo: today the relaxation stack is uniformly rhs-blind, so its McCormick
envelope still relaxes the same row the verifier checks; half-honoured, the envelope
would be built for a *different* row than the one being verified — a soundness
hazard in place of a wrong-answer hazard.

This file tests **both directions** the finding named:

1. the refusal fires, at both doors, for scalar and vector rows, named and unnamed
   (``test_*_refused``);
2. the normalized construction the DSL produces is untouched and still solves to the
   right answer (``test_*_still_solves``) — the control that proves the guard is
   discriminating rather than rejecting everything.

3. the export scoping is right in *both* directions: the writers that honour
   ``rhs`` still emit such a row, the writers that would corrupt it refuse
   (``test_*_writers_*``).

Per CLAUDE.md §6 every arm bumps a module counter and a module-scoped finalizer
fails if this worker executed none. The finalizer deliberately does **not** assert
per-arm totals — CI runs ``-n <workers> --dist loadgroup``, so each worker owns a
separate copy of the counters and a zero there is a statement about the scheduler,
not the code. The complete senses x doors claim is re-derived inside the single
test :func:`test_every_door_refuses_the_full_matrix`, which is therefore always
whole. (Verified by running this file under ``-n 4 --dist loadgroup``, not by
reasoning about it — the sibling Card 4c guard failed in CI on exactly this.)
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from discopt.modeling import Constraint, Model  # noqa: E402

pytestmark = pytest.mark.smoke

# Executed-assertion counters (CLAUDE.md §6). A guard test at the bottom fails if
# any of them is still zero, so a collection error or an over-broad skip cannot
# leave this file reporting "0 violations" and reading as a pass.
COUNTS: dict[str, int] = {
    "refused_subject_to": 0,
    "refused_validate": 0,
    "refused_solve": 0,
    "normalized_solved": 0,
    "public_api_reachable": 0,
    "export_scoped": 0,
}


def _bump(key: str) -> None:
    COUNTS[key] += 1


# ──────────────────────────────────────────────────────────────────────────
# The refusal
# ──────────────────────────────────────────────────────────────────────────


def _scalar_model() -> tuple[Model, object]:
    m = Model("rhs_scalar")
    w = m.continuous("w", lb=0.0, ub=10.0)
    m.minimize(w)
    return m, w


def _vector_model() -> tuple[Model, object]:
    m = Model("rhs_vector")
    x = m.continuous("x", shape=(3,), lb=-10.0, ub=10.0)
    m.minimize(x[0])
    return m, x


@pytest.mark.parametrize("sense", ["<=", ">=", "=="])
@pytest.mark.parametrize("rhs", [5.0, -5.0, 1e-9])
def test_subject_to_refuses_nonzero_rhs(sense, rhs):
    """The public door refuses, on the line that caused it."""
    m, w = _scalar_model()
    with pytest.raises(ValueError, match="non-zero rhs"):
        m.subject_to(Constraint(w, sense, rhs))
    # The refusal must not half-add the row.
    assert m._constraints == []
    _bump("refused_subject_to")


def test_subject_to_list_refuses_nonzero_rhs():
    """The list/iterable arm of ``subject_to`` is a second public door."""
    m, w = _scalar_model()
    with pytest.raises(ValueError, match="non-zero rhs"):
        m.subject_to([w >= 1.0, Constraint(w, ">=", 5.0)], name="rows")
    _bump("refused_subject_to")


@pytest.mark.parametrize("sense", ["<=", ">=", "=="])
def test_validate_refuses_privately_appended_nonzero_rhs(sense):
    """``_constraints.append`` bypasses ``subject_to``; ``validate`` still catches it."""
    m, w = _scalar_model()
    m._constraints.append(Constraint(w, sense, 5.0, "wrow"))
    with pytest.raises(ValueError, match="non-zero rhs"):
        m.validate()
    _bump("refused_validate")


@pytest.mark.parametrize("sense", ["<=", ">="])
def test_solve_refuses_privately_appended_nonzero_rhs(sense):
    """``solve`` always validates, so the wrong answer is now unreachable.

    Before the fix this returned ``optimal`` with ``w = 0`` for ``w >= 5``.
    """
    m, w = _scalar_model()
    m._constraints.append(Constraint(w, sense, 5.0, "wrow"))
    with pytest.raises(ValueError, match="non-zero rhs"):
        m.solve(time_limit=30)
    _bump("refused_solve")


def test_vector_row_with_scalar_rhs_is_refused():
    """A size-3 body against a scalar ``rhs`` — the shape the corpus fixtures use."""
    m, x = _vector_model()
    m._constraints.append(Constraint(x, "<=", 2.0, "xle2"))
    with pytest.raises(ValueError, match="non-zero rhs"):
        m.validate()
    _bump("refused_validate")


def test_vector_rhs_array_is_refused():
    """An array ``rhs`` with any non-zero entry is refused.

    ``np.any`` on the array, not ``float(rhs) != 0`` — the latter raises on a vector
    and would have turned a silent wrong answer into a confusing TypeError.
    """
    m, x = _vector_model()
    m._constraints.append(Constraint(x, "<=", np.array([0.0, 0.0, 3.0]), "xvec"))
    with pytest.raises(ValueError, match="non-zero rhs"):
        m.validate()
    _bump("refused_validate")


def test_refusal_message_is_actionable():
    """The message must name the row, the value, and the rewrite."""
    m, w = _scalar_model()
    m._constraints.append(Constraint(w, ">=", 5.0, "wge5"))
    with pytest.raises(ValueError) as exc:
        m.validate()
    text = str(exc.value)
    for needle in ("wge5", "5.0", "Constraint(body - rhs", "Model.validate"):
        assert needle in text, f"missing {needle!r} from refusal: {text}"
    _bump("refused_validate")


def test_unnamed_row_is_localised_by_index():
    """An unnamed offender still identifies itself."""
    m, w = _scalar_model()
    m._constraints.append(w >= 1.0)  # normalized, fine
    m._constraints.append(Constraint(w, ">=", 5.0))  # offender at index 1
    with pytest.raises(ValueError, match=r"_constraints\[1\]"):
        m.validate()
    _bump("refused_validate")


# ──────────────────────────────────────────────────────────────────────────
# The control: normalized rows are untouched
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("build", "expected"),
    [
        (lambda m, w: m.subject_to(w >= 5.0), 5.0),
        (lambda m, w: m.subject_to(Constraint(5.0 - w, "<=", 0.0)), 5.0),
        (lambda m, w: m.subject_to(w >= 2.5), 2.5),
    ],
    ids=["operator_ge", "explicit_zero_rhs", "operator_ge_frac"],
)
def test_normalized_rows_still_solve(build, expected):
    """The supported construction is unaffected and still gives the right answer.

    Without this arm the refusal could be rejecting every constraint and the file
    would still be green (CLAUDE.md §6 — the discriminator has to discriminate).
    """
    m, w = _scalar_model()
    build(m, w)
    r = m.solve(time_limit=60)
    assert r.status == "optimal", r.status
    assert r.objective == pytest.approx(expected, abs=1e-5)
    _bump("normalized_solved")


def test_operator_form_normalizes_rhs_to_zero():
    """Why the invariant is cheap to keep: the DSL already folds the offset."""
    m, w = _scalar_model()
    m.subject_to(w >= 5.0)
    (c,) = m._constraints
    assert float(c.rhs) == 0.0
    _bump("normalized_solved")


def test_public_api_reach_is_real():
    """``Constraint`` is public, so this was never only a private-attribute bug.

    Recorded as its own assertion because the original write-up concluded "the
    public path is correct" — it is not; ``subject_to`` accepted the object
    verbatim.
    """
    import discopt.modeling as dm

    assert "Constraint" in dm.__all__
    assert dm.Constraint is Constraint
    _bump("public_api_reachable")


# ──────────────────────────────────────────────────────────────────────────
# Export scoping: refuse where the writer would corrupt, allow where it is faithful
# ──────────────────────────────────────────────────────────────────────────


def _model_with_unnormalized_row() -> Model:
    """``w >= 5`` carried as a non-zero ``rhs`` — the row every arm below exports."""
    m = Model("rhs_export")
    w = m.continuous("w", lb=0.0, ub=10.0)
    m.minimize(w)
    m._constraints.append(Constraint(w, ">=", 5.0, "row"))
    return m


@pytest.mark.parametrize(
    ("writer", "needle"),
    [("to_nl", "5.0"), ("to_gams", "=g= 5.0")],
)
def test_faithful_writers_export_an_unnormalized_row(writer, needle):
    """``.nl`` and GAMS honour ``rhs``, so refusing them would block a good export.

    The needle is the *offset itself* appearing in the emitted text: this asserts the
    writer round-tripped the 5.0, not merely that it declined to raise.
    """
    import discopt.export as ex

    text = getattr(ex, writer)(_model_with_unnormalized_row())
    assert needle in text, f"{writer} dropped the rhs: {text}"
    _bump("export_scoped")


@pytest.mark.parametrize("writer", ["to_lp", "to_mps"])
def test_corrupting_writers_refuse_an_unnormalized_row(writer):
    """LP and MPS rebuild the rhs from the body alone, so they must refuse.

    Measured before the fix (CLAUDE.md §11 — this contradicts issue #909's text):
    ``to_lp`` emitted ``c0: w >= 0`` and ``to_mps`` an empty ``RHS`` section for
    this model. A loud refusal is strictly better than a silently wrong file.
    """
    import discopt.export as ex

    with pytest.raises(ValueError, match="non-zero rhs"):
        getattr(ex, writer)(_model_with_unnormalized_row())
    _bump("export_scoped")


def test_for_solve_false_does_not_disable_the_other_validate_checks():
    """The export hatch must narrow to the rhs rule, not switch validation off.

    Otherwise ``for_solve=False`` would be a hole: a missing objective, ``lb > ub``
    and every other invariant would stop being enforced on the export path.
    (Duplicate *constraint* names are deliberately not used here — those only warn,
    per M5/#413.)
    """
    # Missing objective.
    m = Model("no_obj")
    m.continuous("w", lb=0.0, ub=10.0)
    with pytest.raises(ValueError, match="No objective"):
        m.validate(for_solve=False)
    _bump("export_scoped")

    # lb > ub.
    m2 = Model("bad_bounds")
    m2.minimize(m2.continuous("v", lb=5.0, ub=1.0))
    with pytest.raises(ValueError, match="lb > ub"):
        m2.validate(for_solve=False)
    _bump("export_scoped")


def test_every_door_refuses_the_full_matrix():
    """The complete claim, re-derived inside ONE test so xdist cannot fragment it.

    The module counters below live in a worker's own process; under CI's
    ``-n <workers> --dist loadgroup`` the parametrised arms above scatter across
    workers, so no single process sees the whole matrix and the finalizer can only
    prove *this* worker did work. This test therefore re-derives the full
    senses x doors claim itself and asserts the exact expected count — the
    non-vacuous statement, independent of the scheduler. (The lesson is from the
    sibling Card 4c guard, which asserted a cross-worker total and failed in CI on
    a statement about the scheduler rather than about the code.)
    """
    doors = 0
    for sense in ("<=", ">=", "=="):
        for rhs in (5.0, -5.0, 1e-9):
            # door 1: subject_to
            m, w = _scalar_model()
            with pytest.raises(ValueError, match="non-zero rhs"):
                m.subject_to(Constraint(w, sense, rhs))
            doors += 1
            # door 2: validate, after a direct append
            m, w = _scalar_model()
            m._constraints.append(Constraint(w, sense, rhs, "r"))
            with pytest.raises(ValueError, match="non-zero rhs"):
                m.validate()
            doors += 1
    assert doors == 18, doors
    _bump("refused_subject_to")
    _bump("refused_validate")


# ──────────────────────────────────────────────────────────────────────────
# Vacuity guard (CLAUDE.md §6)
# ──────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module", autouse=True)
def _executed_assertion_counts():
    """Fail at module teardown if any arm above executed zero times.

    A module-scoped finalizer rather than a ``test_zzz_`` function on purpose:
    ``pytest-randomly`` shuffles test order, so a last-in-file guard is only a
    guard by luck. Teardown always runs last.

    Scope of the claim, deliberately: this asserts only that **this worker**
    executed real assertions. It cannot assert per-arm totals, because CI runs
    ``-n <workers> --dist loadgroup`` and each worker owns a separate copy of
    ``COUNTS`` — an arm that is zero here may simply have run elsewhere, which is a
    statement about the scheduler, not about the code. The complete
    senses x doors claim is re-derived inside
    :func:`test_every_door_refuses_the_full_matrix`, which is one test and
    therefore always whole.
    """
    yield
    print("\nCard 4c / rhs-refusal executed counts (this worker):")
    for key, n in COUNTS.items():
        print(f"  {key:24s} {n}")
    assert sum(COUNTS.values()) > 0, f"this worker executed no assertions: {COUNTS}"
