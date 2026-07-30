"""Card 4c Task 2 — the vector-constraint corpus the ``.nl`` corpus cannot supply.

Why this file exists
--------------------
Phase 5.5 (``030b44f4``) fixed incumbent verifiers that were **accepting infeasible
points**: the hand-rolled loops advanced one row index per constraint *object* while
``NLPEvaluator.evaluate_constraints`` emits one row per *flat element*, so on a
size-3 vector constraint rows 1 and 2 were never examined and a point violating row
2 by 5.0 came back feasible.

Every row in the in-repo ``.nl`` corpus is **scalar**. The consequence is not that
the fix is unverified — ``test_incumbent_verifier_scale.py`` locks it with unit
tests — but that the *corpus-level* instruments are blind to the class: the
119-instance Regime-N panel and the Card 3c node-tightening parity sweep can never
present a multi-row constraint to a verifier, so a regression that reintroduced
object-indexed rows would pass both. A correctness benchmark over that corpus cannot
see a whole defect class.

This module is that corpus, built through the **modeling API** rather than
``from_nl`` (the ``.nl`` reader emits scalar rows, so it cannot express the class).
It is a plain data module — importable by any suite — deliberately *not* named
``test_*`` so pytest does not collect it directly.

What each case guarantees
-------------------------
Every :class:`VectorCase` carries a model, a point known to be **feasible**, and a
point known to be **infeasible**, plus the flat row the bad point violates. Two
properties are load-bearing and asserted by the consuming suites:

* the bad point is **inside every variable bound and integral where required**, so
  the *only* check that can reject it is the constraint-row check. A verifier that
  gets row alignment wrong therefore cannot be rescued by its bounds check.
* ``violated_flat_row`` is ``> 0`` on the misalignment cases — the violation lives
  on a row a per-*object* index never reaches.

``pre55_row_indexing_accepts`` records, per case, whether the **pre-5.5** row loop
(transcribed verbatim in ``test_vector_constraint_corpus.py``) wrongly accepts the
bad point. It is the non-vacuity evidence CLAUDE.md §6 demands: a coverage claim
whose cases every implementation passes has measured nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["VectorCase", "solvable_vector_cases", "vector_constraint_cases"]


@dataclass(frozen=True)
class VectorCase:
    """One vector-constraint model with a known-good and a known-bad point."""

    name: str
    #: Zero-argument builder. A fresh model per call: evaluator caches key on the
    #: model's structural fingerprint and suites mutate bounds.
    build: object
    feasible: np.ndarray
    infeasible: np.ndarray
    #: Flat evaluator row the infeasible point violates (0-based, over the whole
    #: constraint stream).
    violated_flat_row: int
    #: The label ``verify_point`` must report for that row. Labels are
    #: ``"<name>[<i>]"`` with ``i`` the index *within* the constraint object
    #: (scalar rows carry the bare name), so this is not the flat index — keeping
    #: both is what makes "the violation is on a row a per-object index never
    #: reaches" checkable rather than asserted.
    violated_label: str
    #: Total flat rows the evaluator must emit — the executed-assertion count each
    #: consuming suite checks so a shrunken row map cannot read as a pass.
    n_flat_rows: int
    #: True when the pre-5.5 per-constraint-object row loop wrongly ACCEPTS
    #: ``infeasible``. At least one case must be True or this corpus is vacuous.
    pre55_row_indexing_accepts: bool
    why: str


def _le_tail():
    """A single size-3 ``<=`` vector row with a non-zero scalar ``rhs``."""
    from discopt.modeling.core import Constraint, Model

    m = Model("vec_le_tail")
    x = m.continuous("x", shape=(3,), lb=-10.0, ub=10.0)
    m._constraints.append(Constraint(x, "<=", 2.0, "xle2"))
    m.minimize(x[0])
    return m


def _mixed_senses():
    """Two vector rows of different sense and size, both with non-zero ``rhs``."""
    from discopt.modeling.core import Constraint, Model

    m = Model("vec_mixed_senses")
    x = m.continuous("x", shape=(3,), lb=-10.0, ub=10.0)
    y = m.continuous("y", shape=(2,), lb=-10.0, ub=10.0)
    m._constraints.append(Constraint(x, "<=", 2.0, "xle2"))
    m._constraints.append(Constraint(y, ">=", 1.0, "yge1"))
    m.minimize(x[0] + y[0])
    return m


def _equality():
    """A size-3 vector EQUALITY.

    The offset lives in the *body* (``x - 1``) rather than in ``rhs`` on purpose.
    The pre-5.5 loop carried two independent defects, and with a non-zero ``rhs``
    they entangle: ignoring ``rhs`` makes the old loop reject this point on flat
    row 0 for the *wrong* reason, which would mask the misalignment this case
    exists to isolate. Non-zero ``rhs`` is covered by four other cases; here the
    body is zero at feasibility so row 0 passes the old loop cleanly and the only
    thing left to catch the point is a row the old index never reads.
    """
    from discopt.modeling.core import Constraint, Model

    m = Model("vec_equality")
    x = m.continuous("x", shape=(3,), lb=-10.0, ub=10.0)
    m._constraints.append(Constraint(x - 1.0, "==", 0.0, "xeq1"))
    m.minimize(x[0])
    return m


def _with_integer():
    """A size-3 INTEGER vector row. The bad point is integral and in-bounds, so
    only the row check can reject it."""
    from discopt.modeling.core import Constraint, Model

    m = Model("vec_with_integer")
    n = m.integer("n", shape=(3,), lb=0, ub=5)
    m._constraints.append(Constraint(n, "<=", 2.0, "nle2"))
    m.minimize(n[0])
    return m


def _vector_then_scalar():
    """A size-3 vector row followed by a SCALAR row.

    The per-object index reads the scalar row's verdict off flat row 1 — an
    element of the *vector* row. This is the alignment defect producing a wrong
    answer on a row that is itself perfectly ordinary.
    """
    from discopt.modeling.core import Constraint, Model

    m = Model("vec_then_scalar")
    x = m.continuous("x", shape=(3,), lb=-10.0, ub=10.0)
    w = m.continuous("w", lb=-10.0, ub=10.0)
    m._constraints.append(Constraint(x, "<=", 2.0, "xle2"))
    m._constraints.append(Constraint(w, "<=", -5.0, "wlem5"))
    m.minimize(w)
    return m


def _branching():
    """A vector row on a model that actually **branches**.

    The other cases are root-solvable, which is fine for a verifier but useless to
    the node-tightening parity arm: a stack that never runs cannot be watched. The
    integer vector plus the bilinear row put this model on the spatial / NLP-BB
    path, so ``in_tree_presolve`` and the Python Jacobian pass both see a
    multi-row constraint — the class the ``.nl`` corpus cannot supply.

    **Every offset here lives in the body and every ``rhs`` is 0, and that is
    required, not stylistic.** This is the only case a suite ``solve()``s, and a
    ``Constraint`` appended directly to ``model._constraints`` with a NON-zero
    ``rhs`` is honoured by the evaluator/verifier but *ignored by the solver*:
    measured on this tree, ``Constraint(w, ">=", 5.0)`` appended by hand solves to
    ``w = 0`` while the same row written ``m.subject_to(w >= 5.0)`` (which folds
    the offset into the body) solves to ``w = 5``. A solved case with a non-zero
    ``rhs`` would therefore be a model the solver and the verifier disagree about,
    and any invariant measured across them would be meaningless. The verifier-only
    cases above keep their non-zero ``rhs`` deliberately — exercising it is the
    point there (Phase 5.5 defect 4) and nothing solves them.

    ``test_node_tightening_parity.py`` asserts the all-zero-``rhs`` property before
    solving, so this cannot silently regress.
    """
    from discopt.modeling.core import Constraint, Model

    m = Model("vec_branching")
    n = m.integer("n", shape=(3,), lb=0, ub=4)
    y = m.continuous("y", shape=(3,), lb=0.0, ub=4.0)
    # Scalar row FIRST, vector row second. Ordering matters to what the defect
    # does: with the vector row first, a per-object index reads the scalar row's
    # verdict off ``cap[1]`` and the misalignment produces a wrong *reject* —
    # a defect, but not the wrongly-ACCEPT direction this corpus is built to
    # catch. Scalar-first makes object 0 correctly aligned and leaves the vector
    # row's tail rows unread, which is the accepting failure.
    m._constraints.append(Constraint(n[0] * y[1] + n[1] * y[2] - 3.0, ">=", 0.0, "bil"))
    m._constraints.append(Constraint(n + y - 5.0, "<=", 0.0, "cap"))
    m.minimize(n[0] + n[1] + n[2] + y[0] + y[1] + y[2])
    return m


def _scalar_control():
    """Scalar control. No vector row: a per-object index is CORRECTLY aligned here,
    so this case must NOT be accepted-by-the-old-loop. It proves the discriminator
    in the consuming suite is measuring alignment and not just failing everything.
    """
    from discopt.modeling.core import Constraint, Model

    m = Model("scalar_control")
    w = m.continuous("w", lb=-10.0, ub=10.0)
    m._constraints.append(Constraint(w, "<=", 2.0, "wle2"))
    m.minimize(w)
    return m


def vector_constraint_cases() -> tuple[VectorCase, ...]:
    """The corpus. Ordered vector-first so a failure reads in defect order."""
    f = np.array
    return (
        VectorCase(
            name="vec_le_tail",
            build=_le_tail,
            feasible=f([0.0, 0.0, 1.5]),
            infeasible=f([0.0, 0.0, 7.0]),  # row 2: 7 <= 2 violated by 5.0
            violated_flat_row=2,
            violated_label="xle2[2]",
            n_flat_rows=3,
            pre55_row_indexing_accepts=True,
            why="the violation is on flat row 2; a per-object index only ever reads row 0",
        ),
        VectorCase(
            name="vec_mixed_senses",
            build=_mixed_senses,
            feasible=f([0.0, 0.0, 0.0, 3.0, 3.0]),
            # rows 0-2 (x <= 2) satisfied; row 4 (y[1] >= 1) violated by 1.0.
            infeasible=f([0.0, 0.0, 0.0, 3.0, 0.0]),
            violated_flat_row=4,
            violated_label="yge1[1]",
            n_flat_rows=5,
            pre55_row_indexing_accepts=True,
            why="two objects, five rows: a per-object index reads rows 0 and 1 only",
        ),
        VectorCase(
            name="vec_equality",
            build=_equality,
            feasible=f([1.0, 1.0, 1.0]),
            infeasible=f([1.0, 1.0, 4.0]),  # row 2 residual 3.0
            violated_flat_row=2,
            violated_label="xeq1[2]",
            n_flat_rows=3,
            pre55_row_indexing_accepts=True,
            why="vector equality; the residual sits on a row the old index never reaches",
        ),
        VectorCase(
            name="vec_with_integer",
            build=_with_integer,
            feasible=f([0.0, 0.0, 2.0]),
            # integral and inside [0,5], so bounds and integrality both PASS:
            # only the row check can reject it.
            infeasible=f([0.0, 0.0, 5.0]),
            violated_flat_row=2,
            violated_label="nle2[2]",
            n_flat_rows=3,
            pre55_row_indexing_accepts=True,
            why="integer vector; bad point is integral and in-bounds, so only rows can reject",
        ),
        VectorCase(
            name="vec_then_scalar",
            build=_vector_then_scalar,
            feasible=f([0.0, 0.0, 0.0, -6.0]),
            # x rows all satisfied; the SCALAR row w <= -5 is violated by 4.0.
            infeasible=f([0.0, 0.0, 0.0, -1.0]),
            violated_flat_row=3,
            violated_label="wlem5",
            n_flat_rows=4,
            pre55_row_indexing_accepts=True,
            why="the scalar row's verdict is read off flat row 1, an element of the vector row",
        ),
        VectorCase(
            name="vec_branching",
            build=_branching,
            # n=[2,2,0], y=[2,2,0]: cap = [-1,-1,-5] <= 0; bil = 2*2 + 2*0 = 4 >= 3.
            feasible=f([2.0, 2.0, 0.0, 2.0, 2.0, 0.0]),
            # n=[2,2,4], y=[2,2,4]: cap = [-1,-1,3] — row 2 violated by 3.0, while
            # rows 0/1 pass, every value is integral where required and in-bounds,
            # and the bilinear row (12 >= 3) is satisfied.
            infeasible=f([2.0, 2.0, 4.0, 2.0, 2.0, 4.0]),
            violated_flat_row=3,  # flat stream is [bil, cap[0], cap[1], cap[2]]
            violated_label="cap[2]",
            n_flat_rows=4,
            pre55_row_indexing_accepts=True,
            why="branching model; the violated row is one a per-object index never reads",
        ),
        VectorCase(
            name="scalar_control",
            build=_scalar_control,
            feasible=f([1.0]),
            infeasible=f([7.0]),  # 7 <= 2 violated by 5.0
            violated_flat_row=0,
            violated_label="wle2",
            n_flat_rows=1,
            pre55_row_indexing_accepts=False,
            why="scalar control: a per-object index IS aligned here and must still reject",
        ),
    )


def solvable_vector_cases() -> tuple[VectorCase, ...]:
    """The subset of :func:`vector_constraint_cases` a suite may ``solve()``.

    A ``Constraint`` appended straight onto ``model._constraints`` with a non-zero
    ``rhs`` is honoured by the NLP evaluator (and therefore by every verifier) but
    **ignored by the solver**: measured on this tree, ``Constraint(w, ">=", 5.0)``
    appended by hand solves to ``w = 0``, while ``m.subject_to(w >= 5.0)`` — which
    folds the offset into the body — solves to ``w = 5``.

    That makes such a model a *different problem* to the two consumers, so any
    invariant measured across a solve and a verification of it would be void. The
    non-zero-``rhs`` cases are kept (exercising ``rhs`` is precisely Phase 5.5
    defect 4) and are used **verification-only**; this filter is what keeps them
    out of the solving suites, structurally rather than by convention.

    Filtering is derived from the models themselves, not from a hand-maintained
    list, so a new case joins the right bucket automatically.
    """
    out = []
    for case in vector_constraint_cases():
        model = case.build()
        if all(float(getattr(c, "rhs", 0.0)) == 0.0 for c in model._constraints):
            out.append(case)
    return tuple(out)
