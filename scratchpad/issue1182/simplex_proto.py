"""Prototype of the Wehbeh & Kerrigan (arXiv:2601.03906v1) Theorem-1 lowering.

Entry-experiment code for issue #1182 -- NOT a shipped path. It exists so the
entry experiment CLAUDE.md section 4 demands can be run *before* any production
implementation.

Theorem 1 replaces a CNF clause ``OR_j [p_ij(z) <= 0]`` with

    sum_j lambda_ij * p_ij(z) <= 0,    lambda_i >= 0,    sum_j lambda_ij = 1

which is exact **in projection onto z**: pick ``lambda_i = e_k`` for a satisfied
literal k, and conversely a convex combination of the ``p_ij`` that is <= 0
forces ``min_j p_ij <= 0``.

A discopt disjunction is ``OR_j (AND_k c_jk)``, i.e. a disjunction of
*conjunctions*, so CNF conversion distributes:

    OR_j AND_k p_jk  ==  AND_{(k_1..k_J)} (OR_j p_{j,k_j})

with ``prod_j |P_j|`` clauses. That blowup is one of the four quantities
requirement 4 of #1182 asks to be measured separately, so it is counted here and
never hidden behind "model size".

Each equality ``h == 0`` inside a disjunct is two predicates (``h <= 0`` and
``-h <= 0``), which is where the blowup comes from on the grid-style GDPs.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from discopt.modeling.core import (
    Constraint,
    DisjunctionSemantics,
    Model,
    SelectorActivation,
    _DisjunctiveConstraint,
)

SIMPLEX_AUX_PREFIX = "_simplex_"


class SimplexLoweringRefused(ValueError):
    """The declared semantics or predicate shape is outside Theorem 1's contract."""


@dataclass
class SizeCounts:
    """Requirement 4's four quantities, counted separately (never summed)."""

    disjunctions: int = 0
    cnf_clauses: int = 0
    literal_occurrences: int = 0
    aux_variables: int = 0
    rows: int = 0
    # filled in by the caller from the built model
    jacobian_nonzeros: int = 0

    def add(self, other: "SizeCounts") -> None:
        self.disjunctions += other.disjunctions
        self.cnf_clauses += other.cnf_clauses
        self.literal_occurrences += other.literal_occurrences
        self.aux_variables += other.aux_variables
        self.rows += other.rows


@dataclass
class SourcePredicate:
    """One original predicate, kept so residuals are measured on the *source*.

    Requirement 1 of #1182: the simplex weights are existential witnesses, so the
    truth of the disjunction must be read off the original predicates at the
    returned point, never off the weighted rows and never as a Boolean
    assignment recovered from a fractional lambda.
    """

    disjunction: str
    disjunct: int
    body: object          # Expression p with "p <= 0" meaning the literal is true
    definition: str       # human-readable source of this residual


@dataclass
class LoweredDisjunction:
    name: str
    n_disjuncts: int
    predicates: list[list[SourcePredicate]] = field(default_factory=list)
    counts: SizeCounts = field(default_factory=SizeCounts)


def _predicates_of(con: Constraint, disjunction: str, disjunct: int) -> list[SourcePredicate]:
    """Normalize one disjunct row to a list of ``p <= 0`` predicates."""
    if not isinstance(con, Constraint):
        raise SimplexLoweringRefused(
            f"{disjunction!r} disjunct {disjunct} holds a {type(con).__name__}; "
            "Theorem 1 is stated over algebraic predicates, and a nested "
            "disjunction/SOS row is not one. Refusing rather than approximating."
        )
    if con.rhs != 0.0:
        raise SimplexLoweringRefused(f"{disjunction!r}: non-normalized rhs {con.rhs}")
    if con.sense == "<=":
        return [SourcePredicate(disjunction, disjunct, con.body, "body <= 0")]
    if con.sense == ">=":
        return [SourcePredicate(disjunction, disjunct, -con.body, "-body <= 0")]
    if con.sense == "==":
        return [
            SourcePredicate(disjunction, disjunct, con.body, "body <= 0 (of ==)"),
            SourcePredicate(disjunction, disjunct, -con.body, "-body <= 0 (of ==)"),
        ]
    raise SimplexLoweringRefused(f"{disjunction!r}: unknown sense {con.sense!r}")


def _require_projection_semantics(dc: _DisjunctiveConstraint) -> None:
    sem = getattr(dc, "semantics", DisjunctionSemantics.SELECT_ONE)
    if sem.activation is not SelectorActivation.ONE_WAY:
        raise SimplexLoweringRefused(
            f"{dc.name!r} declares {sem.name}; Theorem 1 is a projection statement "
            "and reproduces ONE_WAY activation only. REIFIED needs the paper's "
            "section 3.1 existential exponential lift for strict negation -- "
            "substituting a closed inequality or a fixed margin changes the "
            "contract (#1182)."
        )


def lower_disjunction(
    dc: _DisjunctiveConstraint,
    model: Model,
    add_weight,
    index: int,
) -> tuple[LoweredDisjunction, list[Constraint]]:
    """Emit Theorem 1's rows for one disjunction. Returns (record, new rows)."""
    _require_projection_semantics(dc)
    name = dc.name or f"disj{index}"
    per_disjunct: list[list[SourcePredicate]] = []
    for j, disjunct in enumerate(dc.disjuncts):
        preds: list[SourcePredicate] = []
        for con in disjunct:
            preds.extend(_predicates_of(con, name, j))
        if not preds:
            raise SimplexLoweringRefused(f"{name!r}: disjunct {j} is empty")
        per_disjunct.append(preds)

    J = len(per_disjunct)
    clauses = list(itertools.product(*per_disjunct))
    counts = SizeCounts(
        disjunctions=1,
        cnf_clauses=len(clauses),
        literal_occurrences=len(clauses) * J,
        aux_variables=len(clauses) * J,
        rows=2 * len(clauses),  # one simplex equality + one weighted row per clause
    )

    rows: list[Constraint] = []
    for i, clause in enumerate(clauses):
        lam = add_weight(f"{name}_c{i}", J)
        # sum_j lambda_ij == 1
        simplex = lam[0]
        for j in range(1, J):
            simplex = simplex + lam[j]
        rows.append(Constraint(body=simplex - 1.0, sense="==", rhs=0.0,
                               name=f"_simplex_{name}_c{i}_sum"))
        # sum_j lambda_ij * p_ij(z) <= 0
        weighted = lam[0] * clause[0].body
        for j in range(1, J):
            weighted = weighted + lam[j] * clause[j].body
        rows.append(Constraint(body=weighted, sense="<=", rhs=0.0,
                               name=f"_simplex_{name}_c{i}_row"))

    return LoweredDisjunction(name, J, per_disjunct, counts), rows


def reformulate_simplex(model: Model) -> tuple[Model, list[LoweredDisjunction], SizeCounts]:
    """Replace every ``_DisjunctiveConstraint`` with Theorem 1's continuous rows."""
    new_model = Model(model.name)
    new_model._variables = list(model._variables)
    new_model._parameters = list(model._parameters)
    new_model._rebuild_name_index()
    new_model._objective = model._objective

    taken = {v.name for v in new_model._variables}
    counter = [0]

    def add_weight(tag: str, size: int):
        name = f"{SIMPLEX_AUX_PREFIX}lam_{counter[0]}"
        counter[0] += 1
        while name in taken:
            name = f"{SIMPLEX_AUX_PREFIX}lam_{counter[0]}"
            counter[0] += 1
        taken.add(name)
        return new_model.continuous(name, size, lb=0.0, ub=1.0)

    records: list[LoweredDisjunction] = []
    total = SizeCounts()
    for idx, c in enumerate(model._constraints):
        if isinstance(c, _DisjunctiveConstraint):
            rec, rows = lower_disjunction(c, new_model, add_weight, idx)
            records.append(rec)
            total.add(rec.counts)
            new_model._constraints.extend(rows)
        else:
            new_model._constraints.append(c)
    return new_model, records, total
