r"""Exact continuous (simplex/CNF) lowering of disjunctions — issue #1182.

Theorem 1 of Wehbeh & Kerrigan, *Exact Continuous Reformulations of Logic
Constraints in Nonlinear Optimization and Optimal Control Problems*
(`arXiv:2601.03906v1 <https://arxiv.org/abs/2601.03906v1>`_), replaces a CNF
clause :math:`\bigvee_j [p_{ij}(z) \le 0]` with

.. math::

    \sum_j \lambda_{ij}\, p_{ij}(z) \le 0, \qquad
    \lambda_i \ge 0, \qquad \sum_j \lambda_{ij} = 1 .

This is exact **in projection onto the original variables** ``z``: pick
:math:`\lambda_i = e_k` for any satisfied literal ``k``, and conversely a convex
combination of the :math:`p_{ij}` that is :math:`\le 0` forces
:math:`\min_j p_{ij} \le 0`. The lifted problem stays nonconvex — exactness is a
statement about the projected feasible set, never about convexity.

A discopt disjunction is :math:`\bigvee_j \bigwedge_k c_{jk}`, so CNF conversion
distributes,

.. math::

    \bigvee_j \bigwedge_k p_{jk}
    \;=\; \bigwedge_{(k_1,\dots,k_J)} \bigl(\textstyle\bigvee_j p_{j,k_j}\bigr),

with :math:`\prod_j |P_j|` clauses. Each equality row ``h == 0`` is two
predicates (``h <= 0`` and ``-h <= 0``), which is where the blowup comes from on
grid-style GDPs — measured, not hidden: :class:`LoweringSizes` reports clauses,
literal occurrences and weight variables **separately**, and a disjunction whose
clause count exceeds :data:`MAX_CNF_CLAUSES` is refused loudly rather than
silently expanded.

Why this is **not** the default (measured, #1182)
-------------------------------------------------
The entry experiment (``scratchpad/issue1182/``) compared this lowering against
big-M and hull, both solved by the same certified global path:

* On the in-repo native GDP corpus (jobshop, ex1_linan_2023, small_batch) it was
  slower on all three and failed to certify ``ex1_linan_2023`` at all within
  60 s, where big-M and hull certify in ~6 s.
* On the paper's own class — obstacle-avoidance optimal control, where the CNF
  distribution is a no-op — it also failed to certify what big-M certifies.

So the *performance* motive for this lowering is falsified for certified global
solving, and ``"simplex"`` is opt-in. What survives, and is why it exists, is a
**capability**: discopt's big-M pass refuses a disjunct row whose interval
enclosure is unbounded, and the Furman–Sawaya–Grossmann hull refuses a row that
is not finite at the origin (:class:`~discopt._relax.gdp_reformulate.
HullPerspectiveOriginError`). Theorem 1 needs neither — its weights are bounded
in ``[0, 1]`` by construction and it forms no perspective. Scanning 11,058
GDPlib disjunct rows found 18 rows (in ``stranded_gas``, ``log`` of a capacity
sum whose box includes 0) that hit **both** refusals, i.e. rows no lowering in
this tree could handle before.

Those two refusals are **not** guards this lowering steps around. Each is a
statement about its own lowering's validity: a fabricated finite ``M``, or a
fabricated ``g(0)``, produces rows that are not the declared feasible set. Theorem
1 fabricates neither quantity because it needs neither. What it hands the
relaxation layer is an ordinary algebraic row — ``sum_j lambda_ij p_ij(z) <= 0``
— of exactly the kind an unlowered model may already contain (discopt accepts
``1/x <= 1`` on a zero-straddling box as a plain constraint today), so the
soundness question it raises is the relaxation layer's usual one and not a new
exemption.

The weights are witnesses, not selectors
----------------------------------------
:math:`\lambda_{ij}` is an *existential witness* for "some literal of clause i
holds". It is a continuous variable in ``[0, 1]``, never a binary, so a
fractional value is not failed Boolean integrality; and it must not be turned
into a named Boolean assignment. The honest answer to "which disjunct holds?" is
:func:`selected_disjuncts`, which reads the **declared source predicates** at the
returned point and can legitimately return more than one disjunct when they
overlap. :func:`disjunction_residuals` likewise measures the source rows, not the
weighted rows, so neither a satisfied weighted row nor a report cached at another
point can stand in for source validation.

Because the lowering is exact in projection, a certified global solve of the
lowered model is a certificate for the source model; this path introduces no
local-only result, so it needs no local/certified status distinction of its own.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

from discopt.modeling.core import (
    GDP_AUX_PREFIX,
    Constraint,
    DisjunctionSemantics,
    Expression,
    Model,
    SelectorActivation,
    Variable,
    _DisjunctiveConstraint,
)

__all__ = [
    "MAX_CNF_CLAUSES",
    "SIMPLEX_WEIGHT_PREFIX",
    "DisjunctionResidual",
    "DisjunctionResidualReport",
    "LoweringSizes",
    "SimplexLoweringRecord",
    "SimplexLoweringRefused",
    "disjunction_residuals",
    "lower_disjunction_simplex",
    "selected_disjuncts",
    "structural_jacobian_nonzeros",
]

#: Name prefix of the Theorem-1 weight variables. They are *witnesses*, not
#: selectors — see the module docstring. Built on ``GDP_AUX_PREFIX`` on purpose:
#: that is the namespace ``Model._check_name`` refuses to user-facing factories
#: (``docs/disjunction_semantics.md`` §4), so a user variable cannot collide with
#: a weight, and the existential-auxiliary status is visible in the name.
SIMPLEX_WEIGHT_PREFIX = f"{GDP_AUX_PREFIX}simplex_lam_"

#: Clause budget per disjunction. CNF distribution is multiplicative in the
#: disjunct sizes, so a disjunction of four 3-row disjuncts is already 81
#: clauses. Exceeding this is refused rather than expanded, because the cost is
#: not visible in the model the user wrote.
MAX_CNF_CLAUSES = 1024


class SimplexLoweringRefused(ValueError):
    """The disjunction is outside Theorem 1's contract, or exceeds the CNF budget.

    A refusal, never a fallback: every case here would otherwise be lowered to
    something that is *not* the declared feasible set (CLAUDE.md §3).
    """


@dataclass
class LoweringSizes:
    """The four size quantities of #1182 requirement 4, counted **separately**.

    They are deliberately not summed into one "model size": the temporal
    constructions of the paper reduce clause counts, and that alone says nothing
    about literal occurrences, weight variables or Jacobian structure.
    """

    disjunctions: int = 0
    cnf_clauses: int = 0
    literal_occurrences: int = 0
    weight_variables: int = 0
    rows: int = 0

    def add(self, other: "LoweringSizes") -> None:
        self.disjunctions += other.disjunctions
        self.cnf_clauses += other.cnf_clauses
        self.literal_occurrences += other.literal_occurrences
        self.weight_variables += other.weight_variables
        self.rows += other.rows


def structural_jacobian_nonzeros(model: Model) -> int:
    """Structural nonzeros of the constraint Jacobian: sum over rows of |vars(row)|.

    The fourth quantity of #1182 requirement 4, and the one that is **not** a
    per-disjunction property, so it lives here rather than on
    :class:`LoweringSizes`: it is a whole-model number and only means anything
    when compared between two lowerings of the same model. Structural, not
    numeric — a coefficient that happens to be zero at some point still occupies
    a Jacobian entry, and the point is to compare sparsity patterns.

    Rows a lowering did not produce (indicator, SOS and logical constraints, and
    any disjunction left unlowered) are **not** silently skipped: they are not
    ``Constraint`` rows, and counting them as zero would understate the pattern,
    so this refuses when it meets one.
    """
    from discopt.modeling.core import _DisjunctiveConstraint as _DC

    nnz = 0
    for index, row in enumerate(model._constraints):
        if isinstance(row, Constraint):
            nnz += len(_variables_in(row.body))
            continue
        kind = "an unlowered disjunction" if isinstance(row, _DC) else type(row).__name__
        raise ValueError(
            f"constraint {index} is {kind}, which has no Jacobian row of its own; "
            "count nonzeros on the LOWERED model, where every row is algebraic, "
            "or the comparison silently omits it"
        )
    return nnz


def _variables_in(expr) -> set[str]:
    """Names of the variables an expression touches (structural, not numeric)."""
    seen: set[str] = set()
    visited: set[int] = set()
    stack = [expr]
    while stack:
        node = stack.pop()
        if id(node) in visited:
            continue
        visited.add(id(node))
        if isinstance(node, Variable):
            seen.add(node.name)
            continue
        for attr in ("operands", "args", "children"):
            kids = getattr(node, attr, None)
            if kids:
                stack.extend(kids)
                break
        else:
            for attr in ("left", "right", "operand", "base", "expr", "body"):
                kid = getattr(node, attr, None)
                if kid is not None and not isinstance(kid, (int, float, str)):
                    stack.append(kid)
    return seen


@dataclass
class SimplexLoweringRecord:
    """What one disjunction was lowered to.

    Carries no selector and no Boolean assignment, by design: ``weight_names``
    names continuous witnesses, and reading a disjunct choice off them is the
    error requirement 1 of #1182 exists to prevent. Use
    :func:`selected_disjuncts`.
    """

    name: str
    n_disjuncts: int
    weight_names: list[str] = field(default_factory=list)
    sizes: LoweringSizes = field(default_factory=LoweringSizes)


@dataclass
class DisjunctionResidual:
    """Source-predicate residual of one disjunction at one point.

    ``violation`` is ``min_j max_k p_jk(z)`` over the **declared** disjunct rows:
    ``<= 0`` exactly when some disjunct holds. ``definition`` travels with the
    number so a residual can never be read as a different quantity.
    """

    name: str
    per_disjunct: list[float]
    violation: float
    definition: str = "min_j max_k p_jk(z) over the declared disjunct rows"


@dataclass
class DisjunctionResidualReport:
    """Residuals for every disjunction of a model at one specific point.

    The point is an argument, not state: there is no cached report, so a report
    taken at another point cannot stand in for source validation (#1182
    requirement 3).
    """

    residuals: list[DisjunctionResidual] = field(default_factory=list)
    comparisons: int = 0

    @property
    def max_violation(self) -> float:
        """Worst source violation over all disjunctions.

        Raises when the report measured nothing, so "no violations" can never be
        the reading of a report that traversed no rows (CLAUDE.md §6).
        """
        if not self.residuals:
            raise ValueError(
                "no disjunction was measured; a max violation over an empty "
                "report would read as a pass"
            )
        return max(r.violation for r in self.residuals)


# ── predicate normalization ──────────────────────────────────────────────────


def _require_scalar(body: Expression, where: str) -> None:
    try:
        shape = body.shape
    except AttributeError:
        # Shape not statically known (reductions, matmul, ...). A reduction is a
        # scalar; anything else would need per-element weights.
        return
    if shape not in ((), (1,)):
        raise SimplexLoweringRefused(
            f"{where}: disjunct row has shape {shape}; Theorem 1 needs one weight "
            "per literal, so a vector row would need per-element weights and a "
            "per-element CNF distribution. Expand the row into scalar rows "
            "(one Constraint per element) or use gdp_method='big-m'/'hull'."
        )


def _predicates_of(con: Constraint, where: str) -> list[Expression]:
    """Normalize one disjunct row to the predicates ``p`` meaning ``p <= 0``."""
    if isinstance(con, _DisjunctiveConstraint):
        raise SimplexLoweringRefused(
            f"{where}: a nested disjunction is not an algebraic predicate. "
            "Theorem 1 is stated over predicates p(z) <= 0; flatten the nesting "
            "or use gdp_method='big-m'."
        )
    if not isinstance(con, Constraint):
        raise SimplexLoweringRefused(
            f"{where}: disjunct row is a {type(con).__name__}, not a Constraint."
        )
    if con.rhs != 0.0:
        raise SimplexLoweringRefused(
            f"{where}: non-normalized rhs {con.rhs!r}; rows must be 'body sense 0'."
        )
    _require_scalar(con.body, where)
    if con.sense == "<=":
        return [con.body]
    if con.sense == ">=":
        return [-con.body]
    if con.sense == "==":
        # An equality is a conjunction of two predicates, and CNF distribution is
        # multiplicative over it — the source of the blowup measured in #1182.
        return [con.body, -con.body]
    raise SimplexLoweringRefused(f"{where}: unknown constraint sense {con.sense!r}")


def _require_projection_semantics(dc: _DisjunctiveConstraint, where: str) -> None:
    semantics = getattr(dc, "semantics", DisjunctionSemantics.SELECT_ONE)
    if semantics.activation is not SelectorActivation.ONE_WAY:
        raise SimplexLoweringRefused(
            f"{where}: declares DisjunctionSemantics.{semantics.name}, whose "
            f"activation is {semantics.activation.name}. Theorem 1 reproduces the "
            "union of the disjuncts, i.e. ONE_WAY activation. REIFIED needs the "
            "paper's §3.1 existential exponential lift for strict negation; "
            "substituting a closed inequality or a fixed margin would change the "
            "declared feasible set, so this is refused rather than approximated."
        )


# ── the lowering ─────────────────────────────────────────────────────────────


def lower_disjunction_simplex(
    dc: _DisjunctiveConstraint,
    add_weight,
    *,
    index: int = 0,
    max_clauses: int = MAX_CNF_CLAUSES,
) -> tuple[SimplexLoweringRecord, list[Constraint]]:
    """Emit Theorem 1's rows for one disjunction.

    Parameters
    ----------
    dc
        The declared disjunction. Its rows are read, never mutated.
    add_weight
        ``add_weight(size) -> Variable`` allocating a continuous weight vector in
        ``[0, 1]`` on the model being built.
    index
        Position of ``dc`` in the source constraint list, used only for naming an
        anonymous disjunction.
    max_clauses
        Per-disjunction CNF budget; exceeding it raises
        :class:`SimplexLoweringRefused`.

    Returns
    -------
    (record, rows)
        ``record`` carries the size counts and the witness names; ``rows`` are
        the constraints to add.
    """
    name = dc.name or f"disj{index}"
    where = f"simplex lowering of {name!r}"
    _require_projection_semantics(dc, where)

    per_disjunct: list[list[Expression]] = []
    for j, disjunct in enumerate(dc.disjuncts):
        preds: list[Expression] = []
        for con in disjunct:
            preds.extend(_predicates_of(con, f"{where}, disjunct {j}"))
        if not preds:
            raise SimplexLoweringRefused(
                f"{where}: disjunct {j} declares no rows, so it is satisfied by "
                "every point and the disjunction is vacuous. Remove it or state "
                "the intended predicate."
            )
        per_disjunct.append(preds)

    n_disjuncts = len(per_disjunct)
    n_clauses = math.prod(len(p) for p in per_disjunct)
    if n_clauses > max_clauses:
        sizes = [len(p) for p in per_disjunct]
        raise SimplexLoweringRefused(
            f"{where}: CNF distribution over disjunct predicate counts {sizes} "
            f"yields {n_clauses} clauses ({n_clauses * n_disjuncts} literal "
            f"occurrences and as many weight variables), above the budget of "
            f"{max_clauses}. Each equality row counts twice. Raise max_clauses "
            "deliberately, or use gdp_method='big-m'/'hull', whose size is "
            "linear in the disjunct rows."
        )

    record = SimplexLoweringRecord(
        name=name,
        n_disjuncts=n_disjuncts,
        sizes=LoweringSizes(
            disjunctions=1,
            cnf_clauses=n_clauses,
            literal_occurrences=n_clauses * n_disjuncts,
            weight_variables=n_clauses * n_disjuncts,
            rows=2 * n_clauses,
        ),
    )

    rows: list[Constraint] = []
    for i, clause in enumerate(itertools.product(*per_disjunct)):
        lam = add_weight(n_disjuncts)
        record.weight_names.append(lam.name)

        simplex_body: Expression = lam[0]
        weighted: Expression = lam[0] * clause[0]
        for j in range(1, n_disjuncts):
            simplex_body = simplex_body + lam[j]
            weighted = weighted + lam[j] * clause[j]

        rows.append(
            Constraint(
                body=simplex_body - 1.0,
                sense="==",
                rhs=0.0,
                name=f"_simplex_{name}_c{i}_weights",
            )
        )
        rows.append(
            Constraint(
                body=weighted,
                sense="<=",
                rhs=0.0,
                name=f"_simplex_{name}_c{i}_clause",
            )
        )
    return record, rows


# ── source-predicate validation (requirement 1) ──────────────────────────────


def _flat_point(model: Model, x_by_name) -> np.ndarray:
    parts = []
    for v in model._variables:
        if x_by_name is None or v.name not in x_by_name:
            raise KeyError(
                f"variable {v.name!r} is absent from the point handed in; a source "
                "residual is measured on the declared operands and cannot be "
                "reconstructed from the lowered rows"
            )
        arr = np.asarray(x_by_name[v.name], dtype=np.float64).reshape(-1)
        if arr.size != v.size:
            raise ValueError(f"{v.name}: expected {v.size} value(s) in the point, got {arr.size}")
        parts.append(arr)
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float64)


def _residual_of(model: Model, dc: _DisjunctiveConstraint, name: str, flat, counter):
    from discopt._relax.dag_compiler import compile_expression

    per_disjunct: list[float] = []
    for j, disjunct in enumerate(dc.disjuncts):
        worst = -math.inf
        for con in disjunct:
            for body in _predicates_of(con, f"residual of {name!r}, disjunct {j}"):
                value = float(np.max(np.asarray(compile_expression(body, model)(flat))))
                worst = max(worst, value)
                counter[0] += 1
        per_disjunct.append(worst)
    return DisjunctionResidual(name=name, per_disjunct=per_disjunct, violation=min(per_disjunct))


def disjunction_residuals(model: Model, point) -> DisjunctionResidualReport:
    """Measure every disjunction of ``model`` at ``point``, on the SOURCE rows.

    ``model`` is the model **as written** — the one still carrying its
    ``either_or`` disjunctions — and ``point`` is a ``{name: value}`` mapping such
    as :attr:`SolveResult.x`. Auxiliary entries in ``point`` (selector binaries,
    hull disaggregates, Theorem-1 weights) are ignored: the residual is a function
    of the declared operands and the source variables only.

    Raises rather than returning an empty report when ``model`` declares no
    disjunction, so "no violations" can never be the reading of a probe that
    traversed nothing (CLAUDE.md §6).
    """
    flat = _flat_point(model, point)
    counter = [0]
    report = DisjunctionResidualReport()
    for index, c in enumerate(model._constraints):
        if isinstance(c, _DisjunctiveConstraint):
            name = c.name or f"disj{index}"
            report.residuals.append(_residual_of(model, c, name, flat, counter))
    report.comparisons = counter[0]
    if not report.residuals:
        raise ValueError(
            "the model declares no disjunction, so this report would measure "
            "nothing; check you passed the model as written, not a lowered copy"
        )
    return report


def selected_disjuncts(model: Model, point, *, tolerance: float = 1e-6) -> dict[str, list[int]]:
    """Which disjuncts actually hold at ``point``, read off the source predicates.

    This is the honest replacement for reading a disjunct choice off a selector:
    under the Theorem-1 lowering there is no selector, and the weights are
    existential witnesses that may be fractional at a perfectly feasible point.

    Returns a mapping from disjunction name to the list of disjunct indices whose
    rows all hold within ``tolerance``. The list may hold **more than one** index
    when the disjuncts overlap — ``SELECT_ONE`` selects one disjunct but does not
    forbid a point from lying in another — and is empty when the disjunction is
    violated at this point, which callers must treat as a failed validation
    rather than as "the first disjunct".
    """
    report = disjunction_residuals(model, point)
    return {
        r.name: [j for j, v in enumerate(r.per_disjunct) if v <= tolerance]
        for r in report.residuals
    }
