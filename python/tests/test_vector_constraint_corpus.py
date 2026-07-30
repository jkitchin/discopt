"""Card 4c Task 2 — exercise the vector-constraint defect class at SUITE level.

Phase 5.5 closed four wrongly-**accept** holes in the incumbent verifiers, one of
which (defect 2, row misalignment) can only manifest on a constraint with more than
one flat row. The in-repo ``.nl`` corpus is entirely scalar, so the 119-instance
Regime-N panel and the Card 3c parity sweep cannot present that class to a verifier:
the fix rests on unit tests alone and a correctness benchmark over the corpus is
blind to it.

``vector_constraint_corpus.py`` supplies the missing class. This file is the guard:

* every case's feasible point verifies and its infeasible point does not, across
  **all three** verifier entry points (the shared verifier and both kernel
  wrappers);
* the infeasible point is proven to be inside every bound and integral, so the row
  check is the only thing that can reject it — a verifier cannot pass these by
  accident;
* :func:`_pre55_verify` is a verbatim transcription of the row loop as it stood at
  ``030b44f4~1`` and is asserted to **wrongly accept** the vector cases. Without
  that arm this file would be a coverage claim nobody had shown was non-vacuous
  (CLAUDE.md §6), and the scalar control proves the arm discriminates rather than
  failing everything.

Per CLAUDE.md §6 the module prints and asserts executed-comparison counts; per §7
nothing here swallows an exception around a comparison.
"""

from __future__ import annotations

import math
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

# ``vector_constraint_corpus`` is a sibling module; pytest's rootdir insertion
# makes it importable, the same way ``relaxation_harness`` is imported elsewhere
# in this directory. No ``sys.path`` surgery — that would shadow for the whole
# session, not just this file.
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from discopt.solver.native_kernel import _native_kernel_verify_point  # noqa: E402
from discopt.solvers._convex_kernel import _incumbent_is_feasible  # noqa: E402
from discopt.validation.feasibility import verify_point  # noqa: E402
from vector_constraint_corpus import vector_constraint_cases  # noqa: E402

# Pinned to one xdist worker. The guard below re-derives the corpus claim itself, so
# the *substantive* assertion is scheduler-independent — but it also asserts that
# THIS worker did real work (`rows_examined > 0` etc.), and under
# `-n <workers> --dist loadgroup` the guard can land on a worker where none of this
# file's other tests ran, zeroing those counters. Caught only by running this file
# together with other files: alone, the scheduler happened to co-locate them.
pytestmark = pytest.mark.xdist_group("vector_constraint_corpus")

CASES = vector_constraint_cases()

#: Module totals, asserted non-zero at the end of the file. A suite that compared
#: nothing must not read as a pass.
TOTALS = {
    "cases": 0,
    "rows_examined": 0,
    "verifier_verdicts": 0,
    "pre55_verdicts": 0,
    "pre55_wrong_accepts": 0,
    "bound_assertions": 0,
    "parity_containment_checks": 0,
}


# --------------------------------------------------------------------------- #
# The pre-5.5 row loop, transcribed verbatim from                              #
# ``git show 030b44f4~1:python/discopt/solver/native_kernel.py``               #
# (``_native_kernel_verify_point``, the ``evaluator.n_constraints > 0`` block).#
#                                                                              #
# Reproduced rather than referenced so the demonstration survives the deletion #
# of the old code, and kept byte-faithful in its three defects:                #
#   * ``idx`` advances once per constraint OBJECT, not per flat row;           #
#   * ``c.rhs`` is ignored (the body is compared against zero);                #
#   * the tolerance is the self-referential ``abs + rel*|val|``.               #
# The bounds/integrality prologue is reproduced too, so when this arm accepts  #
# a point the row loop is demonstrably the only thing that could have caught   #
# it.                                                                          #
# --------------------------------------------------------------------------- #
def _pre55_verify(model, x_flat) -> bool:
    """True iff the pre-Phase-5.5 verifier would have called ``x_flat`` feasible."""
    from discopt.modeling.core import Constraint, VarType

    abs_tol, rel_tol, int_tol = 1e-6, 1e-4, 1e-5
    x_flat = np.asarray(x_flat, dtype=np.float64)
    if not np.all(np.isfinite(x_flat)):
        return False

    off = 0
    for v in model._variables:
        size = int(getattr(v, "size", 1))
        vals = x_flat[off : off + size]
        if vals.shape[0] != size:
            return False
        lb_flat = np.asarray(v.lb, dtype=np.float64).flatten()
        ub_flat = np.asarray(v.ub, dtype=np.float64).flatten()
        lb_tol = abs_tol + rel_tol * np.abs(lb_flat)
        ub_tol = abs_tol + rel_tol * np.abs(ub_flat)
        if np.any(vals < lb_flat - lb_tol) or np.any(vals > ub_flat + ub_tol):
            return False
        if v.var_type in (VarType.INTEGER, VarType.BINARY):
            if np.any(np.abs(vals - np.round(vals)) > int_tol):
                return False
        off += size

    # NOTE: no ``try`` here. The original wrapped this in a bare ``except`` that
    # returned False; swallowing it would hide a broken transcription behind a
    # plausible-looking verdict (CLAUDE.md §7), so let it raise.
    from discopt._jax.nlp_evaluator import cached_evaluator

    evaluator = cached_evaluator(model)
    if evaluator.n_constraints > 0:
        cons = np.asarray(evaluator.evaluate_constraints(x_flat), dtype=np.float64)
        idx = 0
        for c in model._constraints:
            if not isinstance(c, Constraint):
                continue
            if idx >= cons.shape[0]:
                return False
            val = float(cons[idx])
            if not math.isfinite(val):
                return False
            tol = abs_tol + rel_tol * abs(val)
            if c.sense == "<=":
                if val > tol:
                    return False
            elif c.sense == ">=":
                if val < -tol:
                    return False
            elif c.sense == "==":
                if abs(val) > tol:
                    return False
            else:
                return False
            idx += 1  # <-- the defect: one step per OBJECT, not per flat row
    return True


def _all_verifiers(model, x):
    """(shared, native-kernel, convex-kernel) verdicts for one point."""
    x = np.asarray(x, dtype=np.float64)
    shared = verify_point(model, x)
    nk_ok, _ = _native_kernel_verify_point(model, x)
    ck_ok = _incumbent_is_feasible(model, x)
    return shared, bool(nk_ok), bool(ck_ok)


# --------------------------------------------------------------------------- #
# 1. The corpus is well-formed: the bad point can ONLY be caught by a row.     #
# --------------------------------------------------------------------------- #
@pytest.mark.smoke
@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_infeasible_point_is_in_bounds_and_integral(case):
    """The premise the whole file rests on.

    If the bad point were out of bounds or fractional, *every* verifier would
    reject it through its bounds/integrality prologue and the row-alignment arm
    below would prove nothing.
    """
    from discopt.modeling.core import VarType

    model = case.build()
    x = np.asarray(case.infeasible, dtype=np.float64)
    off = 0
    checked = 0
    for v in model._variables:
        size = int(getattr(v, "size", 1))
        vals = x[off : off + size]
        lb = np.asarray(v.lb, dtype=np.float64).flatten()
        ub = np.asarray(v.ub, dtype=np.float64).flatten()
        assert np.all(vals >= lb - 1e-12), f"{case.name}: bad point below lb on {v.name}"
        assert np.all(vals <= ub + 1e-12), f"{case.name}: bad point above ub on {v.name}"
        if v.var_type in (VarType.INTEGER, VarType.BINARY):
            assert np.all(np.abs(vals - np.round(vals)) <= 1e-12), (
                f"{case.name}: bad point is fractional on integer {v.name}"
            )
        checked += size
        off += size
    assert checked == x.size, f"{case.name}: point length {x.size} != {checked} flat columns"
    TOTALS["bound_assertions"] += checked


@pytest.mark.smoke
@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_evaluator_emits_the_expected_flat_rows(case):
    """The evaluator really does emit one row per flat element.

    This is what makes the class a *vector*-constraint class at all; if the row
    map collapsed to one row per object the corpus would silently stop covering
    the defect it exists for.
    """
    from discopt._jax.nlp_evaluator import cached_evaluator

    model = case.build()
    ev = cached_evaluator(model)
    assert ev.n_constraints == case.n_flat_rows, (
        f"{case.name}: evaluator emits {ev.n_constraints} rows, corpus declares "
        f"{case.n_flat_rows} — the corpus no longer covers what it claims"
    )
    rows = np.asarray(ev.evaluate_constraints(case.infeasible), dtype=np.float64)
    assert rows.shape[0] == case.n_flat_rows

    # The label the verifier reports really does sit at `violated_flat_row` in the
    # FLAT stream. Without this the corpus could claim a tail-row violation while
    # actually placing it on row 0, and the whole coverage argument would be air.
    from discopt.validation.feasibility import _row_map

    rmap = _row_map(ev)
    assert rmap is not None, f"{case.name}: the evaluator exposes no row map"
    _senses, _rhs, labels = rmap
    assert len(labels) == case.n_flat_rows
    assert labels[case.violated_flat_row] == case.violated_label, (
        f"{case.name}: flat row {case.violated_flat_row} is labelled "
        f"{labels[case.violated_flat_row]!r}, corpus declares {case.violated_label!r}"
    )

    n_objects = len([c for c in model._constraints])
    if case.pre55_row_indexing_accepts:
        assert n_objects < case.n_flat_rows, (
            f"{case.name}: {n_objects} objects vs {case.n_flat_rows} rows — a "
            "per-object index would be correctly aligned, so this case cannot "
            "demonstrate the misalignment defect"
        )
    TOTALS["rows_examined"] += case.n_flat_rows


# --------------------------------------------------------------------------- #
# 2. The shipped verifiers get every case right, on all three entry points.   #
# --------------------------------------------------------------------------- #
@pytest.mark.smoke
@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_all_verifiers_accept_the_feasible_point(case):
    model = case.build()
    shared, nk_ok, ck_ok = _all_verifiers(model, case.feasible)
    assert shared.n_rows_checked == case.n_flat_rows, (
        f"{case.name}: verifier examined {shared.n_rows_checked} rows, expected "
        f"{case.n_flat_rows} — the probe did not fire on every row"
    )
    assert shared.ok, f"{case.name}: feasible point rejected — {shared.describe()}"
    assert nk_ok, f"{case.name}: native-kernel verifier rejected the feasible point"
    assert ck_ok, f"{case.name}: convex-kernel verifier rejected the feasible point"
    TOTALS["verifier_verdicts"] += 3


@pytest.mark.smoke
@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_all_verifiers_reject_the_infeasible_point(case):
    model = case.build()
    shared, nk_ok, ck_ok = _all_verifiers(model, case.infeasible)
    assert shared.n_rows_checked == case.n_flat_rows, (
        f"{case.name}: verifier examined {shared.n_rows_checked} rows, expected {case.n_flat_rows}"
    )
    assert not shared.ok, f"{case.name}: INFEASIBLE POINT ACCEPTED by the shared verifier"
    assert shared.refusal is None, (
        f"{case.name}: the verifier refused rather than finding the violation — "
        f"{shared.refusal}. A refusal is not the coverage this corpus is for."
    )
    labels = [v.label for v in shared.violations]
    assert case.violated_label in labels, (
        f"{case.name}: expected a violation labelled {case.violated_label!r} "
        f"(flat row {case.violated_flat_row}), got {labels}"
    )
    assert not nk_ok, f"{case.name}: INFEASIBLE POINT ACCEPTED by the native-kernel verifier"
    assert not ck_ok, f"{case.name}: INFEASIBLE POINT ACCEPTED by the convex-kernel verifier"
    TOTALS["verifier_verdicts"] += 3


# --------------------------------------------------------------------------- #
# 3. NON-VACUITY: the pre-5.5 loop wrongly accepts these points.              #
# --------------------------------------------------------------------------- #
@pytest.mark.smoke
@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_pre55_row_indexing_is_what_this_corpus_catches(case):
    """The coverage claim, made falsifiable.

    A corpus that every implementation passes has measured nothing. Each case
    declares whether the pre-Phase-5.5 row loop wrongly ACCEPTS its infeasible
    point, and this asserts that prediction exactly — including the scalar
    control, where the old loop is correctly aligned and must still reject.
    """
    model = case.build()
    old_verdict = _pre55_verify(model, case.infeasible)
    TOTALS["pre55_verdicts"] += 1
    if case.pre55_row_indexing_accepts:
        TOTALS["pre55_wrong_accepts"] += 1
        assert old_verdict is True, (
            f"{case.name}: expected the pre-5.5 loop to WRONGLY ACCEPT this "
            f"infeasible point ({case.why}), but it rejected it. Either the "
            "transcription drifted or the case no longer exercises the defect — "
            "in both cases this corpus has stopped proving what it claims."
        )
    else:
        assert old_verdict is False, (
            f"{case.name}: the scalar control was accepted by the pre-5.5 loop. "
            "The discriminator is broken — it would call any loop defective."
        )
    # And the shipped verifier disagrees with the old one exactly where it should.
    shared = verify_point(model, case.infeasible)
    assert not shared.ok


@pytest.mark.smoke
def test_the_corpus_contains_a_genuine_pre55_failure():
    """File-level: at least one case must actually break the old logic."""
    predicted = [c.name for c in CASES if c.pre55_row_indexing_accepts]
    assert len(predicted) >= 4, (
        f"only {len(predicted)} cases claim to break the pre-5.5 loop: {predicted}"
    )
    assert any(not c.pre55_row_indexing_accepts for c in CASES), (
        "no scalar control — the discriminator is unproven"
    )
    senses = set()
    has_integer = False
    has_nonzero_rhs = False
    has_vector_equality = False
    for c in CASES:
        model = c.build()
        for con in model._constraints:
            senses.add(con.sense)
            if abs(float(con.rhs)) > 0.0:
                has_nonzero_rhs = True
        has_integer |= any(
            getattr(v, "var_type", None) is not None
            and str(getattr(v.var_type, "name", v.var_type)) in ("INTEGER", "BINARY")
            for v in model._variables
        )
        if c.n_flat_rows > 1 and any(con.sense == "==" for con in model._constraints):
            has_vector_equality = True
    # The card names these explicitly: mixed senses, `rhs` set, a vector equality,
    # at least one integer variable.
    assert {"<=", ">=", "=="} <= senses, f"corpus senses {senses} are not mixed"
    assert has_nonzero_rhs, "no case sets a non-zero Constraint.rhs"
    assert has_vector_equality, "no case carries a multi-row equality"
    assert has_integer, "no case carries an integer variable"
    TOTALS["cases"] = len(CASES)


# --------------------------------------------------------------------------- #
# 4. Suite-level totals — a run that compared nothing must fail.              #
# --------------------------------------------------------------------------- #
@pytest.mark.smoke
def test_zz_totals_prove_the_suite_fired():
    """Runs last (``zz`` prefix). CLAUDE.md §6.

    **Self-contained by design.** This guard re-derives the corpus-wide claim itself
    instead of trusting the module-level ``TOTALS`` accumulator to have seen every
    parametrized case. Under ``pytest-xdist`` (CI's PR-fast job runs
    ``-n <workers> --dist loadgroup``) each worker is a separate process with its own
    ``TOTALS``, so an accumulator-only assertion fails with a partial count — 3 of 6
    — which says nothing about the corpus and everything about the scheduler. The
    cases are tiny synthetic models, so recomputing costs milliseconds and holds
    under xdist, ``-k`` filtering, and single-test invocation alike.
    """
    print(f"[vec-corpus] totals (this worker): {TOTALS}")

    # Re-derive: run the pre-5.5 discriminator over every declared case here.
    declared_wrong = [c for c in CASES if c.pre55_row_indexing_accepts]
    recomputed = 0
    for case in CASES:
        model = case.build()
        accepted = _pre55_verify(model, np.asarray(case.infeasible, dtype=np.float64))
        assert accepted is case.pre55_row_indexing_accepts, (
            f"{case.name}: pre-5.5 row loop accepted={accepted}, corpus declares "
            f"{case.pre55_row_indexing_accepts} — the discriminator drifted"
        )
        if accepted:
            recomputed += 1
    assert recomputed == len(declared_wrong), (
        f"{recomputed} wrong-accepts re-derived but {len(declared_wrong)} declared"
    )
    # Measured 2026-07-30: 6 of the 7 cases break the pre-5.5 row loop. The floor
    # is what stops the corpus being whittled down to a vacuous one case.
    assert recomputed >= 6, (
        f"only {recomputed} demonstrated wrong-accepts (was 6); this corpus is no "
        "longer proving the misalignment class is covered"
    )

    # The accumulator still has to show *this* worker did real work — it just no
    # longer has to have seen the whole corpus.
    assert TOTALS["rows_examined"] > 0, "no evaluator rows examined"
    assert TOTALS["verifier_verdicts"] > 0, "no verifier verdicts recorded"
    assert TOTALS["pre55_verdicts"] > 0, "the pre-5.5 discriminator never ran"
    assert TOTALS["bound_assertions"] > 0, "the in-bounds premise was never checked"
