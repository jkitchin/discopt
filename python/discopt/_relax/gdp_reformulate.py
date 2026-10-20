"""GDP (Generalized Disjunctive Programming) reformulation pass.

Converts indicator constraints, disjunctive constraints, and SOS constraints
into standard MINLP constraints via big-M reformulation.

Disjunctions additionally support ``"hull"``, ``"mbigm"``, ``"auto"`` and the
exact *continuous* ``"simplex"`` lowering of #1182 (Theorem 1 of
arXiv:2601.03906v1, implemented in :mod:`discopt._relax.simplex_lowering`).
``"simplex"`` emits no selector binary at all; see that module for why it is
opt-in and which disjunct rows only it can lower.

The reformulation is applied as a preprocessing step before the model is
passed to the NLP evaluator and solver. If no GDP constraints exist, the
original model is returned unchanged (zero overhead).
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import NamedTuple, Optional

import numpy as np

from discopt._relax._numeric import FLOAT_EPS, ROUNDOFF_OPS
from discopt.modeling.core import (
    _BLOCK_ROWS,
    _SELECT_ONE_ROWS,
    GDP_AUX_PREFIX,
    BinaryOp,
    BooleanVar,
    Constant,
    Constraint,
    DisjunctionSemantics,
    Expression,
    FunctionCall,
    IndexExpression,
    LogicalAnd,
    LogicalAtLeast,
    LogicalAtMost,
    LogicalEquivalent,
    LogicalExactly,
    LogicalExpression,
    LogicalImplies,
    LogicalNot,
    LogicalOr,
    Model,
    SelectorActivation,
    SelectorCardinality,
    SumOverExpression,
    UnaryOp,
    Variable,
    VarType,
    _DisjunctiveConstraint,
    _IndicatorConstraint,
    _LogicalConstraint,
    _require_semantics_supported,
    _semantics_supported_by,
    _SOSConstraint,
    _wrap,
    carry_builder_objective,
    carry_validation_guards,
)
from discopt.mpec import carry_complementarities

_DEFAULT_BIG_M = 1e4

# A bound at or above this magnitude is treated as the "unbounded" sentinel, not a
# real finite bound: discopt's unbounded sentinel is 9.999e19 and the NLP path warns
# above ~1e15, so any |bound| >= this is meaningless as a big-M. Using such a value
# as M makes the disjunction link ``operand <= M * b`` satisfiable with the selector
# ``b`` far below the integrality tolerance, silently defeating the disjunction (a
# false optimal). Both big-M paths refuse rather than emit a vacuous M. This is
# orthogonal to the GDP-1/#413 "M too small cuts feasible points" guard.
_BIGM_SENTINEL = 1e15


# What the rows emitted below actually mean. Every reformulation in this module
# emits *one-way* activation (``y_k = 1 => disjunct k holds``) under
# ``sum_k y_k == 1`` — the same pair ``Model.add_disjunction`` emits, so both
# share one declaration and one predicate rather than each spelling out its own
# gate. A semantics is served iff its (activation, cardinality) is what these
# rows encode, so a future member carrying the same pair is supported for free
# and a member carrying a different pair is refused without touching this code.
_SUPPORTED_SEMANTICS: frozenset[tuple[SelectorActivation, SelectorCardinality]] = _SELECT_ONE_ROWS


#: #1775 G1.4: big-M (and its nested form) also encodes OR -- the same one-way
#: activation under ``sum_k y_k >= 1`` (``>= parent`` when nested). Hull, mbigm
#: and the simplex-free paths keep ``_SUPPORTED_SEMANTICS``: Pyomo's hull and
#: mbigm refuse OR too, and a hull's aggregation assumes exactly one selector.
#: One declaration with ``core._BLOCK_ROWS``: block emission and big-M serve the
#: same pairs, so the two gates cannot drift apart.
_BIG_M_SEMANTICS: frozenset[tuple[SelectorActivation, SelectorCardinality]] = _BLOCK_ROWS


def _is_or(dc: _DisjunctiveConstraint) -> bool:
    """True when ``dc`` declares at-least-one (OR) cardinality."""
    return dc.semantics.cardinality is SelectorCardinality.AT_LEAST_ONE


def _semantics_is_supported(semantics: DisjunctionSemantics) -> bool:
    """True when the rows this module emits encode *semantics* exactly."""
    return _semantics_supported_by(semantics, _SUPPORTED_SEMANTICS)


def _require_supported_semantics(
    dc: _DisjunctiveConstraint,
    supported: frozenset[tuple[SelectorActivation, SelectorCardinality]] = _SUPPORTED_SEMANTICS,
) -> None:
    """Refuse any disjunction whose declared semantics these rows do not encode.

    Emitting one-way/exactly-one rows for a disjunction declared ``OR`` or
    ``EXACTLY_ONE_TRUE`` would silently hand back a different feasible set than
    the model declares — a truth-semantics model would be solved as select-one
    and certified. Refuse loudly instead (issue #1124). ``supported`` is what the
    calling lowering emits (big-M passes :data:`_BIG_M_SEMANTICS`).
    """
    semantics = getattr(dc, "semantics", DisjunctionSemantics.SELECT_ONE)
    _require_semantics_supported(semantics, supported, f"disjunction {dc.name or '<anon>'!r}")


#: Every ``method`` :func:`reformulate_gdp` implements. Anything else is refused:
#: the dispatch below routes an unrecognised name to big-M, so before this check a
#: typo (or a not-yet-implemented native name such as ``"gloa"``) solved as big-M
#: and reported ``optimal`` with no warning.
GDP_REFORMULATION_METHODS: frozenset[str] = frozenset({"big-m", "hull", "mbigm", "auto", "simplex"})


def _require_reformulation_method(method: object, what: str) -> None:
    if method not in GDP_REFORMULATION_METHODS:
        allowed = ", ".join(sorted(repr(m) for m in GDP_REFORMULATION_METHODS))
        raise ValueError(f"Unknown {what} {method!r}. Choose one of: {allowed}.")


def reformulate_gdp(
    model: Model,
    method: str = "big-m",
    *,
    respect_disjunction_methods: bool = True,
) -> Model:
    """Replace GDP constraints with standard MINLP constraints.

    Parameters
    ----------
    model : Model
        Input model potentially containing indicator, disjunctive, or SOS
        constraints.
    method : str, default "big-m"
        Reformulation method for disjunctive constraints:
        ``"big-m"`` (default), ``"hull"`` (convex hull),
        ``"mbigm"`` (multiple big-M with LP-based tightening),
        ``"auto"`` to dispatch per-disjunction via the F1 advisor in
        :mod:`discopt._relax.gdp_advisor` (each disjunction independently
        gets the structurally-best of the three), or ``"simplex"`` for the exact
        continuous CNF lowering of #1182
        (:mod:`discopt._relax.simplex_lowering`), which introduces continuous
        weights in ``[0, 1]`` instead of selector binaries. ``"simplex"`` is
        measurably slower than big-M/hull on the corpora benchmarked in #1182
        and exists for the disjunct rows neither of them can lower at all
        (unbounded interval enclosure *and* non-finite at the origin); it is
        never selected by ``"auto"``.
        Indicator and SOS constraints always use big-M regardless.
    respect_disjunction_methods : bool, default True
        Whether a disjunction-local ``method`` override should take precedence
        over the solver-wide ``method``. Set to ``False`` for solver frontends
        that need a specific GDP lowering, such as AMP's MILP relaxations.

    Returns
    -------
    Model
        A new model with GDP constraints replaced by equivalent standard
        constraints. If no GDP constraints exist, returns the original
        model unchanged.
    """
    _require_reformulation_method(method, "GDP reformulation method")
    has_gdp = any(
        isinstance(
            c,
            (_IndicatorConstraint, _DisjunctiveConstraint, _SOSConstraint, _LogicalConstraint),
        )
        for c in model._constraints
    )
    if not has_gdp:
        return model

    new_model = Model(model.name)
    # Copy variables (share the same Variable objects so expressions still work)
    new_model._variables = list(model._variables)
    new_model._parameters = list(model._parameters)
    new_model._rebuild_name_index()  # keep the name cache in sync (M7)
    new_model._objective = model._objective
    # #1798: fast-API rows live only in the source's builder; the loop below walks
    # ``_constraints`` and would drop them. (The fast-API objective is re-issued at
    # the end, once every auxiliary column exists.)
    new_model._constraints.extend(model._builder_linear_constraints())

    # Track auxiliary binaries added by the reformulation
    _aux_counter = [0]

    # Names already live in the model being built. Generated auxiliaries are
    # allocated against this set so a selector can never reuse a name that
    # already exists — the counter alone does not guarantee that, since the
    # pass may run over a model that already carries auxiliaries (a second
    # lowering, an imported model). ``Model._check_name`` keeps *user* names
    # out of the reserved prefix; this keeps generated names unique among
    # themselves.
    _taken: set[str] = {v.name for v in new_model._variables}

    def _add_aux_binary(prefix: str, shape=()) -> Variable:
        name = f"{GDP_AUX_PREFIX}{prefix}_{_aux_counter[0]}"
        _aux_counter[0] += 1
        while name in _taken or name in new_model._names:
            name = f"{GDP_AUX_PREFIX}{prefix}_{_aux_counter[0]}"
            _aux_counter[0] += 1
        _taken.add(name)
        var = Variable(
            name,
            VarType.BINARY,
            shape if isinstance(shape, tuple) else (shape,),
            0.0,
            1.0,
            new_model,
        )
        new_model._register_generated_variable(var)  # #1799
        return var

    _weight_counter = [0]

    def _add_simplex_weight(size: int) -> Variable:
        """Allocate one Theorem-1 weight vector: continuous, in ``[0, 1]``.

        Continuous **on purpose**: these are existential witnesses, not
        selectors, so a fractional value at a feasible point is not a failed
        Boolean integrality (#1182 requirement 1).
        """
        from discopt._relax.simplex_lowering import SIMPLEX_WEIGHT_PREFIX

        name = f"{SIMPLEX_WEIGHT_PREFIX}{_weight_counter[0]}"
        _weight_counter[0] += 1
        while name in _taken or name in new_model._names:
            name = f"{SIMPLEX_WEIGHT_PREFIX}{_weight_counter[0]}"
            _weight_counter[0] += 1
        _taken.add(name)
        var = Variable(name, VarType.CONTINUOUS, (size,), 0.0, 1.0, new_model)
        new_model._register_generated_variable(var)  # #1799
        return var

    # #1615 C-10a: under ``gdp_method="hull"`` the Disjunct blocks of
    # ``Model.add_disjunction`` (indicator rows tagged with their disjunction) are
    # lowered by hull, with the user's indicators as the selectors -- the model
    # already carries their ``exactly(1)`` row. ``_hull_groups`` maps a group key
    # to its rebuilt ``_DisjunctiveConstraint`` and selectors; empty when no group
    # qualifies. Blocks and disjunctions nested in a hulled group are hulled inside
    # it (#1775 G2.1) and named in ``_hull_plan``'s absorbed sets.
    _hull_plan = (
        _collect_indicator_hull_groups(model)
        if method == "hull"
        else _HullPlan({}, {}, set(), set())
    )
    _hull_groups = _hull_plan.groups

    def _hulled_inside(c) -> bool:
        if isinstance(c, _IndicatorConstraint):
            return c.disjunction is not None and c.disjunction[0] in _hull_plan.absorbed_keys
        return id(c) in _hull_plan.absorbed_ids

    # Every other indicator row -- a plain ``Model.if_then``, any block under
    # ``"simplex"`` -- is lowered by big-M (or multiple big-M). Say so rather than
    # let a caller believe ``gdp_method`` reached it.
    if method in ("hull", "simplex"):
        _n_ind = sum(
            1
            for c in model._constraints
            if not _hulled_inside(c)
            and (
                (
                    isinstance(c, _IndicatorConstraint)
                    and (c.disjunction is None or c.disjunction[0] not in _hull_groups)
                )
                or (isinstance(c, _DisjunctiveConstraint) and c.parent_indicator is not None)
            )
        )
        if _n_ind:
            _which = (
                "Model.if_then rows, add_disjunction blocks the hull cannot take "
                "(an empty disjunct, a shared indicator, an OR block, which may "
                "select several disjuncts, or a block nested in one of those), and "
                "OR disjunctions nested in a Disjunct"
                if method == "hull"
                else "Model.if_then, and the Disjunct blocks of Model.add_disjunction"
            )
            warnings.warn(
                f"gdp_method={method!r} does not apply to the {_n_ind} indicator "
                f"constraint(s) in this model ({_which}): they are reformulated by "
                f"big-M. Model.either_or disjunctions take the {method!r} "
                "reformulation"
                + (", as do complete add_disjunction blocks." if method == "hull" else "."),
                UserWarning,
                stacklevel=2,
            )

    # In auto mode, ask the F1 advisor for a per-disjunction
    # recommendation up front; if any disjunction (or indicator) is
    # routed to mbigm we still need the LP relaxation precomputed.
    auto_advice: dict[int, str] = {}
    if method == "auto":
        from discopt._relax.gdp_advisor import recommend_methods

        for adv in recommend_methods(model):
            auto_advice[adv.disjunction_index] = adv.recommendation

    # Precompute LP relaxation for MBM (or auto mode that may pick mbigm).
    lp_data = None
    if method == "mbigm" or "mbigm" in auto_advice.values():
        lp_data = _precompute_lp_relaxation(model)

    # #1775 G2.3 (flag DISCOPT_MBIGM_CONDITIONED, default OFF): under "mbigm", a
    # SELECT_ONE disjunction without nesting -- an either_or, or a complete
    # add_disjunction block group -- gets M conditioned on each other disjunct.
    # Anything the conditioned route declines keeps today's route.
    _mbc_on = lp_data is not None and _mbigm_conditioned_enabled()
    _mbc_groups: dict[str, tuple[_DisjunctiveConstraint, list[Variable]]] = {}
    if _mbc_on and method == "mbigm":
        _mbc_groups = {
            key: (dc, sels)
            for key, (dc, sels) in _collect_indicator_hull_groups(model).groups.items()
            if not any(isinstance(it, _DisjunctiveConstraint) for d in dc.disjuncts for it in d)
        }
    _mbc_done: dict[str, Optional[list[Constraint]]] = {}
    _mbc_budget = [_MBIGM_CONDITIONED_LP_BUDGET]  # shared by the whole lowering

    _hull_emitted: set[str] = set()
    for ci, c in enumerate(model._constraints):
        if _hulled_inside(c):
            continue  # lowered inside its parent's hull (#1775 G2.1)
        if (
            isinstance(c, _IndicatorConstraint)
            and c.disjunction is not None
            and c.disjunction[0] in _mbc_groups
        ):
            key = c.disjunction[0]
            if key not in _mbc_done:
                dc, sels = _mbc_groups[key]
                res = _reformulate_disjunction_mbigm_conditioned(
                    dc, new_model, _add_aux_binary, lp_data, selectors=sels, budget=_mbc_budget
                )
                _mbc_done[key] = None if res is None else res[1]
                if res is not None:
                    new_model._constraints.extend(res[1])
            if _mbc_done[key] is not None:
                continue  # the whole group was lowered above
            # declined: this row takes today's route below
        if (
            isinstance(c, _IndicatorConstraint)
            and c.disjunction is not None
            and c.disjunction[0] in _hull_groups
        ):
            key = c.disjunction[0]
            if key not in _hull_emitted:
                _hull_emitted.add(key)
                dc, sels = _hull_groups[key]
                new_vars, new_cons = _reformulate_disjunction_hull(
                    dc,
                    new_model,
                    _add_aux_binary,
                    selectors=sels,
                    child_selectors=_hull_plan.child_selectors,
                )
                new_model._constraints.extend(new_cons)
        elif isinstance(c, _IndicatorConstraint):
            new_cons = _reformulate_indicator(c, new_model, lp_data=lp_data)
            new_model._constraints.extend(new_cons)
        elif isinstance(c, _DisjunctiveConstraint) and c.parent_indicator is not None:
            # #1775 G1.2: nested in a Disjunct block. The nested big-M lowering ties
            # its selectors to the block's indicator (they sum to it), whatever the
            # method, unless the hull took it in with its parent (#1775 G2.1, skipped
            # above): the simplex lowering has no selectors, and a hull of the child
            # alone would assume its parent selected. Counted in the warning above
            # for those two methods.
            new_vars, new_cons = _reformulate_disjunction_nested(
                c, new_model, _add_aux_binary, parent_selector=c.parent_indicator
            )
            new_model._constraints.extend(new_cons)
        elif isinstance(c, _DisjunctiveConstraint):
            # A per-disjunction ``method`` override (set e.g. by Model.if_else)
            # takes precedence over the solver-wide method / auto advice.
            override = getattr(c, "method", None) if respect_disjunction_methods else None
            if override is not None:
                _require_reformulation_method(
                    override, f"method on disjunction {c.name or '<anon>'!r}"
                )
                chosen = override
            else:
                chosen = auto_advice.get(ci, method) if method == "auto" else method
            # #1775 G1.1: indicators the user read are the selectors. ``None`` (never
            # read) is each lowering's default, so it generates its own exactly as
            # before.
            own_sels = list(c._indicator_vars) if c._indicator_vars is not None else None
            if chosen == "simplex":
                if own_sels is not None:
                    raise ValueError(
                        f"gdp_method='simplex' cannot lower disjunction "
                        f"{c.name or '<anon>'!r}: its indicators were read (to be "
                        "referenced in logic, fixed or priced), and the simplex "
                        "lowering has no selector binaries to bind them to. Use "
                        + ("'big-m'." if _is_or(c) else "'big-m', 'mbigm' or 'hull'.")
                    )
                new_vars, new_cons = _reformulate_disjunction_simplex(
                    c, new_model, _add_simplex_weight, ci
                )
            elif chosen == "hull":
                new_vars, new_cons = _reformulate_disjunction_hull(
                    c, new_model, _add_aux_binary, selectors=own_sels
                )
            else:
                # big-m and mbigm both go through _reformulate_disjunction;
                # the difference is only whether lp_data is supplied.
                if chosen == "mbigm":
                    # #1775 G1.4: OR is a big-M feature only; mbigm refuses it,
                    # as Pyomo's gdp.mbigm does.
                    _require_supported_semantics(c)
                disj_lp = lp_data if chosen == "mbigm" else None
                res = (
                    _reformulate_disjunction_mbigm_conditioned(
                        c,
                        new_model,
                        _add_aux_binary,
                        lp_data,
                        selectors=own_sels,
                        budget=_mbc_budget,
                    )
                    if chosen == "mbigm" and _mbc_on
                    else None
                )
                if res is not None:
                    new_vars, new_cons = res
                else:
                    new_vars, new_cons = _reformulate_disjunction(
                        c, new_model, _add_aux_binary, lp_data=disj_lp, selectors=own_sels
                    )
            new_model._constraints.extend(new_cons)
        elif isinstance(c, _SOSConstraint):
            new_cons = _reformulate_sos(c, new_model, _add_aux_binary)
            new_model._constraints.extend(new_cons)
        elif isinstance(c, _LogicalConstraint):
            new_cons = _logical_to_linear(c, new_model, _add_aux_binary)
            new_model._constraints.extend(new_cons)
        else:
            # Regular Constraint -- keep as-is
            new_model._constraints.append(c)

    # A hulled block none of whose disjuncts has a row of its own (each holds only
    # nesting) tags no row, so the loop above never reached it: lower it here, or
    # the nested disjunctions it took in would be dropped (#1790 shapes, G2.1).
    for key, (dc, sels) in _hull_groups.items():
        if key not in _hull_emitted:
            _hull_emitted.add(key)
            new_vars, new_cons = _reformulate_disjunction_hull(
                dc,
                new_model,
                _add_aux_binary,
                selectors=sels,
                child_selectors=_hull_plan.child_selectors,
            )
            new_model._constraints.extend(new_cons)

    # Complementarity provenance (#1147). The rows above materialize a declared
    # ``0 <= f _|_ g >= 0`` into a selector binary and big-M/hull rows; the
    # *relation* — its source operands, its bounds, its scale — lives only on
    # ``_complementarities`` and was dropped here, leaving the pair name baked
    # into generated identifiers as the sole surviving trace. Forward it.
    carry_complementarities(model, new_model, pass_name="gdp lowering")
    carry_validation_guards(model, new_model)  # #1498
    carry_builder_objective(model, new_model)  # #1798; after the last new column

    return new_model


class _HullPlan(NamedTuple):
    """What :func:`_collect_indicator_hull_groups` hands the GDP pass.

    ``groups`` maps the key of each top-level ``add_disjunction`` block the hull
    lowers to its rebuilt disjunction and selectors. ``child_selectors`` maps
    ``id`` of a block nested inside one of them to its indicators, whose select row
    (``sum == parent``) the model already carries. ``absorbed_keys`` /
    ``absorbed_ids`` name the nested blocks and the nested ``Model.disjunction``
    objects lowered *inside* a group, so the pass skips them where they stand.
    """

    groups: dict[str, tuple[_DisjunctiveConstraint, list[Variable]]]
    child_selectors: dict[int, list[Variable]]
    absorbed_keys: set[str]
    absorbed_ids: set[int]


def _collect_indicator_hull_groups(model: Model) -> _HullPlan:
    """Rebuild the ``Model.add_disjunction`` blocks the hull can lower (#1615 C-10a).

    ``add_disjunction`` lowers each Disjunct to ``if_then`` rows tagged
    ``(key, k, n)``. A group qualifies when it is SELECT_ONE, every disjunct
    ``0..n-1`` has content, each disjunct's rows share one indicator with
    ``active_value == 1``, and the ``n`` indicators are distinct. An empty
    disjunct leaves its indicator unrecorded, and the hull needs every selector,
    so such a group stays on big-M (and is counted in the caller's warning). An OR
    block may select several disjuncts, and the hull's aggregation assumes exactly
    one, so it stays on big-M too (#1775 G1.4).

    Nesting (#1775 G2.1). A block nested in a Disjunct sums its indicators to the
    parent's, not to 1, so it cannot be hulled on its own: with the parent off, its
    hull would force every disaggregated copy -- and so every shared variable -- to
    0. It is hulled *inside* its parent instead, as an item of the parent disjunct,
    when the parent group is itself hulled; so is a ``Model.disjunction`` passed to
    ``Disjunct.subject_to``. A disjunct whose only content is nesting has its
    indicator from the nesting record. Whatever is not taken in stays on big-M,
    tied to its parent's indicator exactly as before.
    """
    rows: dict[str, dict[int, list[_IndicatorConstraint]]] = {}
    sizes: dict[str, int] = {}
    parents: dict[str, Optional[Variable]] = {}
    excluded: set[str] = set()
    for c in model._constraints:
        if isinstance(c, _IndicatorConstraint) and c.disjunction is not None:
            key, k, n = c.disjunction
            rows.setdefault(key, {}).setdefault(k, []).append(c)
            if key in parents and parents[key] is not c.parent:
                excluded.add(key)  # inconsistent nesting tag: leave it on big-M
            parents.setdefault(key, c.parent)
            if c.semantics is not DisjunctionSemantics.SELECT_ONE:
                excluded.add(key)
            if sizes.setdefault(key, n) != n:
                sizes[key] = -1  # inconsistent tag: leave the group on big-M
    members = getattr(model, "_gdp_block_members", {})
    for key, mem in members.items():
        if key in rows and parents.get(key) is not mem.parent:
            excluded.add(key)
        parents.setdefault(key, mem.parent)
        if mem.semantics is not DisjunctionSemantics.SELECT_ONE:
            excluded.add(key)
        if sizes.setdefault(key, len(mem.indicators)) != len(mem.indicators):
            sizes[key] = -1

    # The nested content of each disjunct, keyed by the id of its indicator.
    child_keys: dict[int, list[str]] = {}
    for key, par in parents.items():
        if par is not None:
            child_keys.setdefault(id(par), []).append(key)
    child_dcs: dict[int, list[_DisjunctiveConstraint]] = {}
    for c in model._constraints:
        if isinstance(c, _DisjunctiveConstraint) and c.parent_indicator is not None:
            child_dcs.setdefault(id(c.parent_indicator), []).append(c)

    def _selectors(key: str) -> Optional[list[Variable]]:
        n = sizes.get(key, -1)
        if key in excluded or n < 1:
            return None
        by_k = rows.get(key, {})
        mem = members.get(key)
        sels: list[Variable] = []
        for k in range(n):
            ks = by_k.get(k, [])
            inds = {id(r.indicator) for r in ks}
            if len(inds) > 1 or any(r.active_value != 1 for r in ks):
                return None
            if ks:
                ind = ks[0].indicator
                if mem is not None and mem.indicators[k] is not ind:
                    return None
            elif mem is not None:
                ind = mem.indicators[k]
                if id(ind) not in child_keys and id(ind) not in child_dcs:
                    return None  # an empty disjunct: no content to hull
            else:
                return None
            if not isinstance(ind, Variable):
                return None
            sels.append(ind)
        if len({id(v) for v in sels}) != n:
            return None
        return sels

    child_selectors: dict[int, list[Variable]] = {}
    absorbed_keys: set[str] = set()
    absorbed_ids: set[int] = set()

    def _build(key: str, sels: list[Variable]) -> _DisjunctiveConstraint:
        by_k = rows.get(key, {})
        disjuncts: list[list] = []
        for k, ind in enumerate(sels):
            items: list = [r.constraint for r in by_k.get(k, [])]
            for ck in child_keys.get(id(ind), []):
                csels = _selectors(ck)
                if csels is None:
                    continue  # stays on big-M, tied to ``ind``
                child = _build(ck, csels)
                child_selectors[id(child)] = csels
                absorbed_keys.add(ck)
                items.append(child)
            for cdc in child_dcs.get(id(ind), []):
                if cdc.semantics is not DisjunctionSemantics.SELECT_ONE:
                    continue  # OR: big-M only
                absorbed_ids.add(id(cdc))
                items.append(cdc)
            disjuncts.append(items)
        return _DisjunctiveConstraint(disjuncts=disjuncts, name=key.replace(":", "_"))

    groups: dict[str, tuple[_DisjunctiveConstraint, list[Variable]]] = {}
    for key in parents:
        if parents[key] is not None:
            continue
        sels = _selectors(key)
        if sels is None:
            continue
        groups[key] = (_build(key, sels), sels)
    return _HullPlan(groups, child_selectors, absorbed_keys, absorbed_ids)


#: Real-valued functions undefined on a whole *region* of the line, as
#: ``name -> (test(lo, hi) -> True when [lo, hi] reaches into that region, domain)``.
#: Only a region counts: a boundary point or a pole (``log`` at exactly 0, a
#: divisor reaching 0, ``tan``) is left alone -- see :func:`_domain_gap`.
_RESTRICTED_DOMAINS: dict[str, tuple[Callable[[float, float], bool], str]] = {
    "sqrt": (lambda lo, hi: lo < 0.0, "x >= 0"),
    "log": (lambda lo, hi: lo < 0.0, "x > 0"),
    "log2": (lambda lo, hi: lo < 0.0, "x > 0"),
    "log10": (lambda lo, hi: lo < 0.0, "x > 0"),
    "log1p": (lambda lo, hi: lo < -1.0, "x > -1"),
    "entropy": (lambda lo, hi: lo < 0.0, "x >= 0"),
    "asin": (lambda lo, hi: lo < -1.0 or hi > 1.0, "-1 <= x <= 1"),
    "acos": (lambda lo, hi: lo < -1.0 or hi > 1.0, "-1 <= x <= 1"),
    "acosh": (lambda lo, hi: lo < 1.0, "x >= 1"),
    "atanh": (lambda lo, hi: lo < -1.0 or hi > 1.0, "-1 < x < 1"),
}


def _domain_gap(expr: Expression, model: Model) -> Optional[str]:
    """Where ``expr`` is undefined on a region of the variable box, or ``None`` (#1779).

    Every node is judged against its own domain over the interval enclosure of
    its operands: a function from :data:`_RESTRICTED_DOMAINS`, or a power whose
    base can go negative under a non-integer or variable exponent.

    Judging each node, not the row's enclosure, is the point: an outer operation
    can return a finite enclosure over an undefined inner one -- ``sqrt`` clips
    its argument to ``[0, inf)``, ``log``'s upper end stays finite, and
    ``exp(-abs(x**0.5))`` folds the ``(-inf, inf)`` of an undefined power back to
    ``(0, 1]`` -- and a finite enclosure is exactly what lets a big-M be built.

    **Regions only, not points.** A lowering that keeps the body where its
    disjunct is inactive imposes the body's domain on every other disjunct; that
    cuts their feasible points when the undefined set is a region (``sqrt`` of a
    negative: the measured false optimum of #1779). An undefined *point* -- ``log``
    or a negative power at exactly 0, a divisor reaching 0 -- is approached from
    inside the domain to within the solver's tolerance, which is exactly the
    capability the simplex lowering exists for (#1182: ``1/x <= 1`` with ``x``
    straddling 0, the ``stranded_gas`` capacity rows with ``log`` of a sum that
    may be 0). Measured on ``origin/main`` before this check: simplex is correct
    on those point gaps -- including an "off" disjunct sitting exactly on the
    point -- and wrong on the region gap. Big-M needs no help with point gaps:
    their enclosures are already unbounded where it matters, and it refuses them.

    Node types with no known domain (``CustomCall``, unlisted functions) are not
    judged here; their enclosure is ``(-inf, inf)`` unless proven otherwise.
    """
    from discopt.modeling.core import Parameter

    memo = _BoundMemo()
    _seed_self_products(expr, model, memo)
    seen: set[int] = set()
    stack: list = [expr]
    while stack:
        node = stack.pop()
        if id(node) in seen or not isinstance(node, Expression):
            continue
        seen.add(id(node))
        if isinstance(node, FunctionCall):
            stack.extend(node.args)
            rule = _RESTRICTED_DOMAINS.get(node.func_name)
            if rule is not None and node.args:
                lo, hi = _bound_expression(node.args[0], model, memo)
                if rule[0](lo, hi):
                    return (
                        f"{node.func_name}() is defined only for {rule[1]}, but its "
                        f"argument ranges over [{lo:.6g}, {hi:.6g}]"
                    )
        elif isinstance(node, BinaryOp):
            stack.extend((node.left, node.right))
            if node.op == "**":
                lo, hi = _bound_expression(node.left, model, memo)
                if lo < 0.0:
                    # A Parameter exponent is judged at its current value: the
                    # lowering runs again on every solve, after any change to it.
                    if isinstance(node.right, (Constant, Parameter)):
                        exps = np.ravel(np.asarray(node.right.value, dtype=np.float64))
                    else:
                        return (
                            "a power with a variable exponent needs a nonnegative "
                            f"base, but its base ranges over [{lo:.6g}, {hi:.6g}]"
                        )
                    frac = [p for p in exps if not (np.isfinite(p) and p == int(p))]
                    if frac:
                        return (
                            f"x**{frac[0]:g} is real only for x >= 0, but its base "
                            f"ranges over [{lo:.6g}, {hi:.6g}]"
                        )
        else:
            stack.extend(_expr_children(node))
    return None


def _expr_children(node) -> list:
    """The operands of an expression node, for the generic walks below."""
    out: list = []
    for attr in ("operand", "base", "left", "right"):
        child = getattr(node, attr, None)
        if child is not None:
            out.append(child)
    for attr in ("args", "terms"):
        out.extend(getattr(node, attr, None) or ())
    return out


def _seed_self_products(expr: Expression, model: Model, memo: _BoundMemo) -> None:
    """Enclose every ``a * a`` in ``expr`` as a square, in ``memo``, bottom-up.

    Plain interval multiplication treats the two factors as independent, so
    ``x * x`` on ``x in [-1, 10]`` encloses as ``[-10, 100]``; that made
    :func:`_domain_gap` refuse ``sqrt(x*x + 1)`` and the product-written distance
    rows of layout models, which are defined on the whole box (review of #1787).
    A product of two structurally equal operands -- the same object, or the same
    printed expression, which names every leaf -- is a square, so ``[0, max^2]``
    when the operand's enclosure straddles 0. Seeded only into this check's memo:
    the big-M enclosure itself is unchanged, so no M moves.
    """
    order: list = []
    seen: set[int] = set()
    stack: list = [(expr, False)]
    while stack:
        node, done = stack.pop()
        if done:
            order.append(node)
            continue
        if id(node) in seen or not isinstance(node, Expression):
            continue
        seen.add(id(node))
        stack.append((node, True))
        stack.extend((c, False) for c in _expr_children(node))
    for node in order:  # children before parents
        if (
            isinstance(node, BinaryOp)
            and node.op == "*"
            and (node.left is node.right or str(node.left) == str(node.right))
        ):
            lo, hi = _bound_expression(node.left, model, memo)
            sq = sorted((lo * lo, hi * hi))
            memo.put_bound(node, (0.0 if lo <= 0.0 <= hi else sq[0], sq[1]))


def _domain_gap_error(constraint: Constraint, gap: str, how: str) -> ValueError:
    """The loud refusal for a disjunct/indicator row undefined on part of the box."""
    cname = getattr(constraint, "name", None) or "<unnamed>"
    return ValueError(
        f"cannot lower constraint {cname!r} by {how}: it is undefined on part of the "
        f"variable box ({gap}). {how.capitalize()} keeps the row's body in the model "
        "even where its disjunct (or indicator) is inactive, so the solver would "
        "impose the body's domain on every disjunct and cut feasible points of the "
        "others -- a false certificate (#1779). Bound the variable(s) so the body is "
        "defined on the whole box (e.g. lb=0 for sqrt), or use gdp_method='hull' for "
        "an either_or / add_disjunction disjunction, which keeps the body on the "
        "disjunct's own copy of the variables."
    )


def _compute_big_m(
    constraint: Constraint,
    model: Model,
    default: float = _DEFAULT_BIG_M,
) -> float:
    """Compute a big-M bound for a constraint from variable bounds.

    For a constraint ``body sense 0``, we need an upper bound on ``body``
    (for ``<=``), a lower bound (for ``>=``), or both (for ``==``).

    Uses interval arithmetic over the expression tree to get the tightest
    valid M. **Correctness rule (GDP-1, #413):** M *must* be a valid
    over-estimate of the body over the true variable domain — a big-M that is
    too small turns the inactive disjunct's relaxed constraint into a real
    constraint and can cut feasible points of the *active* disjunct, producing
    a false-infeasible / wrong-optimum certificate. Therefore:

    - When the relevant body bound is **finite and below the ``_BIGM_SENTINEL``
      magnitude**, that bound *is* a valid M and is used as-is. A large M weakens
      the LP relaxation (a *performance* cost), but correctness comes first — we
      never shrink such a valid finite M to a smaller ``default``.
    - When the relevant body bound is **non-finite (``±inf``) or at/above the
      ``_BIGM_SENTINEL`` "unbounded" magnitude** (e.g. a variable left at the
      default ±9.999e19 sentinel), no *usable* finite M exists, so we **refuse
      loudly** rather than emit a vacuous M. A sentinel-sized M is not merely weak:
      it makes the disjunction link ``operand <= M * selector`` hold with the
      selector binary far below the integrality tolerance, so the disjunction is
      never enforced and the solver can certify a point that violates it (a false
      optimal — see GDP-2). The caller must supply a finite bound on the offending
      variable, or use ``method='hull'`` (no big-M). This is *orthogonal* to the
      GDP-1/#413 "M too small cuts feasible points" guard, which still holds.

    Parameters
    ----------
    constraint : Constraint
        The constraint whose body expression needs a big-M.
    model : Model
        Model containing variable bound information.
    default : float
        Fallback M used only when the body is provably bounded by ``default``
        (i.e. the interval bound is finite and no tighter value is available).
        It is never used to *replace* a larger finite bound, and never used to
        paper over an infinite bound.

    Returns
    -------
    float
        A valid (over-estimating) big-M value.

    Raises
    ------
    ValueError
        If the constraint body is unbounded in the direction the big-M must
        cover (no valid finite M exists).
    """
    gap = _domain_gap(constraint.body, model)
    if gap is not None:
        raise _domain_gap_error(constraint, gap, "big-M")
    lo, hi = _bound_expression(constraint.body, model)

    def _require_finite_upper() -> float:
        # Valid M for a ``body <= 0`` disjunct: any value >= max body. The
        # interval upper bound ``hi`` is exactly that. Finite AND below the
        # unbounded sentinel ⇒ valid; non-finite or sentinel-sized ⇒ no *usable*
        # finite M exists (a sentinel-sized M makes the disjunction vacuous).
        if np.isfinite(hi) and abs(hi) < _BIGM_SENTINEL:
            return abs(hi)
        raise _unbounded_big_m_error(constraint, "above")

    def _require_finite_lower() -> float:
        # Valid M for a ``body >= 0`` disjunct: any value >= -min(body) = |lo|.
        if np.isfinite(lo) and abs(lo) < _BIGM_SENTINEL:
            return abs(lo)
        raise _unbounded_big_m_error(constraint, "below")

    if constraint.sense == "<=":
        M = _require_finite_upper()
    elif constraint.sense == ">=":
        M = _require_finite_lower()
    elif constraint.sense == "==":
        # Need to cover both directions.
        M = max(_require_finite_upper(), _require_finite_lower())
    else:
        M = default

    # Ensure M is positive and add small safety margin. ``default`` is a floor
    # (a body whose valid bound is tiny still needs a nonzero M), never a cap.
    return max(abs(M), 1e-8) * 1.01


def _unbounded_big_m_error(constraint: Constraint, direction: str) -> ValueError:
    """Build the loud-refusal error for an unbounded big-M (GDP-1, #413)."""
    cname = getattr(constraint, "name", None) or "<unnamed>"
    return ValueError(
        f"GDP big-M reformulation cannot bound the body of constraint "
        f"{cname!r} from {direction}: its interval enclosure is unbounded, so "
        f"no valid finite big-M exists. A too-small big-M would cut feasible "
        f"points of the active disjunct and yield a false certificate. Add a "
        f"finite bound to the variable(s) in this disjunct/indicator (e.g. "
        f"m.continuous(..., lb=..., ub=...)). The 'hull' reformulation "
        f"(method='hull') needs no big-M, but it too needs a finite bound on every "
        f"variable of the disjunction, within each disjunct (#1617)."
    )


def _compute_big_m_lp(
    constraint: Constraint,
    model: Model,
    lp_data: tuple,
    default: float = _DEFAULT_BIG_M,
) -> float:
    """Compute big-M via LP solve for tighter bounds (Multiple Big-M).

    Solves LP relaxation to maximize/minimize the constraint body expression.
    Falls back to interval arithmetic when the body is nonlinear or LP fails.

    Parameters
    ----------
    constraint : Constraint
        The constraint whose body needs a big-M.
    model : Model
        Model containing variable information.
    lp_data : tuple
        Precomputed LP data: (A_ub, b_ub, A_eq, b_eq, bounds, n_vars).
    default : float
        Fallback M value.
    """
    # Big-M from an LP optimum must use an EXACT (simplex/vertex) oracle, never
    # the POUNCE IPM: the LP optimum becomes the disjunct's big-M, so an inexact
    # optimum (the IPM's analytic-center objective can be grossly wrong on an
    # ill-conditioned LP while still reporting OPTIMAL — issue #145) can return a
    # big-M that is *too small* and cut off feasible points of the inactive
    # disjunct. When no exact oracle is available, fall back to interval
    # arithmetic, which always over-estimates M and is therefore sound.
    try:
        from discopt.solvers import SolveStatus
        from discopt.solvers.lp_backend import get_exact_lp_solver

        solve_lp = get_exact_lp_solver()
    except ImportError:
        solve_lp = None
    if solve_lp is None:
        return _compute_big_m(constraint, model, default)

    A_ub, b_ub, A_eq, b_eq, n_vars = lp_data

    # Extract linear coefficients from constraint body
    # Reuse the same coefficient extraction logic from obbt
    coeffs = _extract_body_coeffs(constraint.body, model, n_vars)
    if coeffs is None:
        # Nonlinear body: fall back to interval arithmetic
        return _compute_big_m(constraint, model, default)

    c_vec, offset = coeffs

    # Build variable bounds, one entry per *flat scalar element* (aligned with
    # the column offsets produced by `_extract_body_coeffs`, which are
    # cumulative `v.size` + flat index). X-2 (#413): an array variable block is
    # NOT a scalar — the previous seed read element 0's bounds
    # (`v.lb.flat[0]`/`v.ub.flat[0]`) and stamped them onto every element. On
    # heterogeneous per-element bounds that shrinks the LP box for elements
    # whose true bounds are wider than element 0's, so the LP-optimum big-M can
    # come back *too small* and cut off feasible points of the inactive
    # disjunct — the exact unsoundness the "exact oracle only" comment above
    # guards against. Read each element's own bound.
    #
    # GDP-2 (#413): ``lp_data`` (its ``c_vec``/``A_ub``/``A_eq`` and ``n_vars``)
    # is precomputed from the ORIGINAL model, before the reformulation appends
    # selector/aux binaries. But ``model`` here is the *mutated* ``new_model``
    # that already carries those aux vars, so iterating all of
    # ``model._variables`` yields MORE scalar columns than ``n_vars``. Passing a
    # ``bounds`` list longer than ``c``/``A`` desynchronizes the LP dimensions
    # and crashes the exact LP backend (``copy_from_slice`` length mismatch),
    # breaking every mbigm disjunction that adds a selector. The aux vars are
    # always appended *after* the originals, so the first ``n_vars`` scalar
    # columns are exactly the original variables that ``c_vec`` indexes — take
    # only those.
    bounds_list = _lp_box(model, n_vars)

    _INF_THRESH = _BIGM_SENTINEL  # shared "unbounded" magnitude; see _compute_big_m

    def _solve_bound(maximize: bool) -> float | None:
        """Solve LP to get bound. Returns None if LP fails."""
        obj = -c_vec if maximize else c_vec
        # #1520: no except. The exact LP oracle (``lp_simplex.solve_lp``) declines
        # by status, handled here; a raise is a defect -- e.g. the ``copy_from_slice``
        # length mismatch described above, which this handler used to turn into
        # "use the default big-M" instead of a bug report.
        result = solve_lp(
            c=obj,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds_list,
        )
        if result.status == SolveStatus.OPTIMAL and result.objective is not None:
            val = -result.objective if maximize else result.objective
            return float(val) + offset
        return None

    if constraint.sense == "<=":
        hi = _solve_bound(maximize=True)
        if hi is not None and abs(hi) < _INF_THRESH:
            return max(abs(hi), 1e-8) * 1.01
    elif constraint.sense == ">=":
        lo = _solve_bound(maximize=False)
        if lo is not None and abs(lo) < _INF_THRESH:
            return max(abs(-lo), 1e-8) * 1.01
    elif constraint.sense == "==":
        hi = _solve_bound(maximize=True)
        lo = _solve_bound(maximize=False)
        if hi is not None and lo is not None:
            M = max(abs(hi), abs(lo))
            if M < _INF_THRESH:
                return max(M, 1e-8) * 1.01

    # Fall back to interval arithmetic
    return _compute_big_m(constraint, model, default)


def _lp_box(model: Model, n_vars: int) -> list[tuple[float, float]]:
    """The LP's variable bounds: one ``(lb, ub)`` per flat scalar column, for the
    first ``n_vars`` columns (the original variables ``lp_data`` indexes; see
    :func:`_compute_big_m_lp` on X-2 and GDP-2, #413)."""
    bounds_list: list[tuple[float, float]] = []
    for v in model._variables:
        if len(bounds_list) >= n_vars:
            break
        lb_flat = np.asarray(v.lb, dtype=np.float64).ravel()
        ub_flat = np.asarray(v.ub, dtype=np.float64).ravel()
        for i in range(v.size):
            if len(bounds_list) >= n_vars:
                break
            lb_i = float(lb_flat[i]) if lb_flat.size == v.size else float(lb_flat.flat[0])
            ub_i = float(ub_flat[i]) if ub_flat.size == v.size else float(ub_flat.flat[0])
            bounds_list.append((lb_i, ub_i))
    return bounds_list


#: Exact LPs one lowering may spend on conditioned Ms (#1775 G2.3). Sizing found
#: ~0.1 s per LP on batch_processing (450 LPs, 41 s) and >5,000 on kaibel; past the
#: budget the remaining rows keep today's unconditioned M, so a large model cannot
#: spend its whole time limit before branch-and-bound starts.
_MBIGM_CONDITIONED_LP_BUDGET = 1000

#: Smallest conditioned M, relative to today's, short of dropping the term
#: (#1824 review): a valid M only has to be ``>=`` the rigorous LP bound, and a
#: coefficient within 1e3 of today's keeps the matrix's range near today's.
_MBIGM_CONDITIONED_MIN_REL = 1e-3


def _mbigm_conditioned_enabled() -> bool:
    """Is #1775 G2.3's conditioned mbigm on? (``DISCOPT_MBIGM_CONDITIONED``, default OFF)."""
    from discopt.solver_tuning import current as _tuning

    return bool(_tuning().mbigm_conditioned)


def _conditioned_ms(
    con: Constraint,
    k: int,
    linear_rows: list[tuple[list, list, list, list]],
    model: Model,
    lp_data: tuple,
    box: list[tuple[float, float]],
    solve_lp,
    budget: Optional[list[int]] = None,
) -> tuple[list[float], list[float]]:
    """Per-other-disjunct big-Ms of row ``con`` of disjunct ``k`` (#1775 G2.3).

    Returns ``(up, down)``, one entry per disjunct (``0.0`` at ``k``): ``up[j]``
    bounds ``body`` from above and ``down[j]`` bounds ``-body``, over the global
    linear rows, disjunct ``j``'s linear rows and the box. Dropping ``j``'s
    nonlinear rows only enlarges the LP region, so each M stays valid.

    Every M starts at today's unconditioned value (:func:`_compute_big_m_lp`,
    which also refuses an unbounded or undefined body exactly as before) and is
    lowered only by a *rigorous* upper bound ``safe`` on the LP maximum
    (:func:`_safe_lp_max`, the Neumaier-Shcherbina bound read off the OPTIMAL
    vertex's duals), never by the raw vertex objective: a float simplex optimum
    can undershoot the true maximum by an amount set by its dual-feasibility
    tolerance times the box widths, which no fixed tolerance bounds. With ``safe``:

    * ``safe <= 0`` -- every point of disjunct ``j``'s LP region has ``body <= 0``,
      so disjunct ``j`` implies the row and the term is dropped (``M = 0``). This
      removes no point at all, so no LP accuracy enters the argument.
    * ``safe > 0`` -- ``M = max(1.01 * safe, _MBIGM_CONDITIONED_MIN_REL * M_now)``
      (capped at today's M): valid because it is ``>= safe``, and never a tiny
      coefficient. A tiny one is what broke med_term_purchasing: a ``1e-6`` floor
      put 940 such coefficients into it and HiGHS's root LP on that matrix failed
      discopt's certificate check (primal/dual pair dual-infeasible).

    The first version snapped ``M`` to 0 whenever the raw objective was below
    ``1e-9 * max(1, |offset|, M_now)``. That tolerance grows with the box, so on
    ``x, z in [0, 1e6]`` it dropped a term whose true maximum was ``1e-4`` and
    certified a wrong optimum (#1824 review). A body or rhs that is not a single
    scalar keeps today's M (see :func:`_is_scalar_linear_row`).

    A status that is not OPTIMAL -- INFEASIBLE included -- or an OPTIMAL exit
    without finite duals keeps today's M: an LP-infeasible disjunct could justify
    ``M = 0`` (it can never be selected), but only on a checked certificate, which
    this route does not take. ``budget`` is a one-element list counting the exact
    LPs this lowering may still run; at 0 the remaining rows keep today's M
    (deterministic, unlike a wall-clock cap).
    """
    from discopt.solvers import SolveStatus

    M_now = _compute_big_m_lp(con, model, lp_data)
    n = len(linear_rows)
    up = [0.0 if j == k else M_now for j in range(n)]
    down = list(up)
    A_ub, b_ub, A_eq, b_eq, n_vars = lp_data
    if not _is_scalar_linear_row(con):
        return up, down  # array-valued body or rhs: today's M for every j
    coeffs = _extract_body_coeffs(con.body, model, n_vars)
    if coeffs is None:
        return up, down  # nonlinear body: today's M for every j
    c_vec, offset = coeffs
    want_up = con.sense in ("<=", "==")
    want_down = con.sense in (">=", "==")
    for j, (ub_rows, ub_rhs, eq_rows, eq_rhs) in enumerate(linear_rows):
        if j == k:
            continue
        Au, bu = _stack_rows(A_ub, b_ub, ub_rows, ub_rhs)
        Ae, be = _stack_rows(A_eq, b_eq, eq_rows, eq_rhs)
        for side, wanted, sign in (("up", want_up, -1.0), ("down", want_down, 1.0)):
            if not wanted:
                continue
            if budget is not None:
                if budget[0] <= 0:
                    continue  # budget spent: today's M
                budget[0] -= 1
            # up: max body = -min(-c.x); down: max(-body) = -min(c.x) - offset.
            res = solve_lp(c=sign * c_vec, A_ub=Au, b_ub=bu, A_eq=Ae, b_eq=be, bounds=box)
            if res.status != SolveStatus.OPTIMAL or res.objective is None:
                continue
            safe = _safe_lp_max(sign * c_vec, res, Au, bu, Ae, be, box)
            if safe is None:
                continue  # no rigorous bound: today's M
            safe += offset if side == "up" else -offset
            safe += 4.0 * FLOAT_EPS * abs(offset)  # the rounding of that addition
            if not np.isfinite(safe) or abs(safe) >= _BIGM_SENTINEL:
                continue
            if safe <= 0.0:
                M_j = 0.0  # body <= safe <= 0 on all of disjunct j: the row is implied
            else:
                M_j = max(1.01 * safe, _MBIGM_CONDITIONED_MIN_REL * M_now)
            if side == "up":
                up[j] = min(up[j], M_j)
            else:
                down[j] = min(down[j], M_j)
    return up, down


def _safe_lp_max(f, res, A_ub, b_ub, A_eq, b_eq, box) -> Optional[float]:
    """Rigorous upper bound on ``max -f.x`` (= ``-min f.x``) over the LP that
    ``res`` solved, or ``None`` when none can be formed (#1824 review).

    ``-g`` for the Neumaier-Shcherbina lower bound ``g <= min f.x``
    (:func:`discopt._relax.obbt._ns_safe_lp_lower_bound`, which subtracts its own
    float-evaluation margin): valid for any multipliers, so the simplex's
    tolerances cannot make it undershoot. Also never below the vertex value, which
    is a feasible point's objective up to primal tolerance."""
    import scipy.sparse as sp

    from discopt._relax.obbt import _ns_safe_lp_lower_bound

    parts_A = [sp.csr_matrix(a) for a in (A_ub, A_eq) if a is not None]
    parts_b = [np.asarray(v, dtype=np.float64).ravel() for v in (b_ub, b_eq) if v is not None]
    A = sp.vstack(parts_A).tocsr() if parts_A else None
    b = np.concatenate(parts_b) if parts_b else None
    n_eq = 0 if A_eq is None else int(sp.csr_matrix(A_eq).shape[0])
    lo = np.array([lb for lb, _ in box], dtype=np.float64)
    hi = np.array([ub for _, ub in box], dtype=np.float64)
    g = _ns_safe_lp_lower_bound(f, getattr(res, "dual_values", None), A, b, lo, hi, n_eq=n_eq)
    if g is None or not np.isfinite(g):
        return None
    return max(-float(g), -float(res.objective))


def _is_scalar_linear_row(con: Constraint) -> bool:
    """Is ``con`` one scalar row as :func:`_extract_body_coeffs` reads it?

    That walker sums an array-valued ``Constant``/``Parameter`` into one offset, so
    ``x >= np.array([5, 1])`` (body ``Constant((2,)) - x``) reads as ``6 - 2x <=
    0``-like nonsense (#1825). An array-valued variable already declines there
    (``resolve_scalar_slot``); this rejects the array-valued leaves it does not."""
    from discopt.modeling.core import BinaryOp, Constant, Parameter, SumOverExpression, UnaryOp

    if np.size(con.rhs) != 1:
        return False
    stack: list = [con.body]
    while stack:
        e = stack.pop()
        if isinstance(e, (Constant, Parameter)):
            if np.size(e.value) != 1:
                return False
        elif isinstance(e, BinaryOp):
            stack.extend((e.left, e.right))
        elif isinstance(e, UnaryOp):
            stack.append(e.operand)
        elif isinstance(e, SumOverExpression):
            stack.extend(e.terms)
    return True


def _stack_rows(A, b, rows: list, rhs: list):
    """``A``/``b`` (global, possibly ``None``) with extra dense rows appended."""
    import scipy.sparse as sp

    if not rows:
        return A, b
    extra = sp.csr_matrix(np.asarray(rows, dtype=np.float64))
    A2 = extra if A is None else sp.vstack([sp.csr_matrix(A), extra]).tocsr()
    b2 = np.asarray(rhs, dtype=np.float64)
    if b is not None:
        b2 = np.concatenate([np.asarray(b, dtype=np.float64).ravel(), b2])
    return A2, b2


def _disjunct_linear_rows(disjunct: list, model: Model, n_vars: int):
    """A disjunct's linear rows as ``(A_ub, b_ub, A_eq, b_eq)`` lists, ``<=`` form."""
    ub_rows: list = []
    ub_rhs: list = []
    eq_rows: list = []
    eq_rhs: list = []
    for con in disjunct:
        if isinstance(con, _DisjunctiveConstraint) or not _is_scalar_linear_row(con):
            continue  # omitting a row only enlarges j's LP region, so every M stays valid
        coeffs = _extract_body_coeffs(con.body, model, n_vars)
        if coeffs is None:
            continue
        a, off = coeffs
        rhs = float(np.asarray(con.rhs).ravel()[0]) - off
        if con.sense == "<=":
            ub_rows.append(a)
            ub_rhs.append(rhs)
        elif con.sense == ">=":
            ub_rows.append(-a)
            ub_rhs.append(-rhs)
        elif con.sense == "==":
            eq_rows.append(a)
            eq_rhs.append(rhs)
    return ub_rows, ub_rhs, eq_rows, eq_rhs


def _reformulate_disjunction_mbigm_conditioned(
    dc: _DisjunctiveConstraint,
    model: Model,
    add_aux_binary,
    lp_data: Optional[tuple],
    selectors: Optional[list[Variable]] = None,
    budget: Optional[list[int]] = None,
) -> tuple[list[Variable], list[Constraint]] | None:
    """Multiple big-M conditioned on each disjunct (#1775 G2.3, flag-gated).

    A row ``body <= 0`` of disjunct ``k`` becomes ``body <= sum_{j != k} M_{c,j}
    y_j`` (``>=`` and ``==`` likewise), with the ``M_{c,j}`` of
    :func:`_conditioned_ms`. With exactly one ``y_j = 1`` the row is ``body <= 0``
    when ``k`` is selected and ``body <= M_{c,j}`` -- a valid bound over disjunct
    ``j``'s region -- otherwise: exact at every integer point. Each ``M_{c,j}`` is
    at most today's M, so the relaxation is at least as tight as unconditioned
    mbigm.

    SELECT_ONE disjunctions without nesting only; returns ``None`` (the caller
    keeps today's route) for anything else, when there is no LP relaxation, or
    when no exact LP oracle exists.
    ``selectors`` as in :func:`_reformulate_disjunction`: given ones are the
    caller's, whose select row the model already carries. ``budget`` is the
    lowering's shared LP budget (see :func:`_conditioned_ms`).
    """
    if lp_data is None:
        return None  # no LP relaxation to condition on
    items = [it for d in dc.disjuncts for it in d]
    if _is_or(dc) or any(isinstance(it, _DisjunctiveConstraint) for it in items):
        return None
    # Big-M bounds ``body`` against 0 (operator-built rows always have rhs 0); a
    # row with a nonzero rhs keeps today's route rather than a new convention.
    if any(np.any(np.asarray(it.rhs, dtype=np.float64) != 0.0) for it in items):
        return None
    try:
        from discopt.solvers.lp_backend import get_exact_lp_solver

        solve_lp = get_exact_lp_solver()
    except ImportError:
        return None
    if solve_lp is None:
        return None
    _require_supported_semantics(dc)
    n_disjuncts = len(dc.disjuncts)
    new_vars: list[Variable] = []
    new_cons: list[Constraint] = []
    if selectors is not None:
        if len(selectors) != n_disjuncts:
            raise ValueError(
                f"mbigm of disjunction {dc.name or 'anon'!r}: {len(selectors)} selectors "
                f"for {n_disjuncts} disjuncts."
            )
        selectors = list(selectors)
    else:
        selectors = []
        for k in range(n_disjuncts):
            y_k = add_aux_binary(f"disj_{dc.name or 'anon'}_{k}")
            selectors.append(y_k)
            new_vars.append(y_k)
        sum_expr: Expression = selectors[0]
        for k in range(1, n_disjuncts):
            sum_expr = sum_expr + selectors[k]
        new_cons.append(
            Constraint(
                body=sum_expr - _wrap(1.0),
                sense="==",
                rhs=0.0,
                name=f"_gdp_select_{dc.name}" if dc.name else None,
            )
        )

    n_vars = lp_data[4]
    box = _lp_box(model, n_vars)
    if budget is None:
        budget = [_MBIGM_CONDITIONED_LP_BUDGET]
    linear_rows = [_disjunct_linear_rows(d, model, n_vars) for d in dc.disjuncts]
    for k, disjunct in enumerate(dc.disjuncts):
        for j, con in enumerate(disjunct):
            up, down = _conditioned_ms(con, k, linear_rows, model, lp_data, box, solve_lp, budget)
            cname = f"_gdp_mbc_{dc.name}_d{k}_c{j}" if dc.name else None

            def _relax(ms: list[float]) -> Expression:
                terms = [_wrap(m) * selectors[i] for i, m in enumerate(ms) if i != k and m != 0.0]
                out: Expression = _wrap(0.0)
                for t in terms:
                    out = out + t
                return out

            if con.sense in ("<=", "=="):
                new_cons.append(
                    Constraint(
                        body=con.body - _relax(up),
                        sense="<=",
                        rhs=0.0,
                        name=(cname + "_le") if cname and con.sense == "==" else cname,
                    )
                )
            if con.sense in (">=", "=="):
                new_cons.append(
                    Constraint(
                        body=con.body + _relax(down),
                        sense=">=",
                        rhs=0.0,
                        name=(cname + "_ge") if cname and con.sense == "==" else cname,
                    )
                )
    return new_vars, new_cons


def _extract_body_coeffs(
    expr: Expression,
    model: Model,
    n_vars: int,
) -> tuple[np.ndarray, float] | None:
    """Extract linear coefficient vector from an expression.

    Returns (c_vec, offset) where body(x) = c_vec @ x + offset,
    or None if the expression is nonlinear or is not a single scalar row
    (an array-valued variable, index or constant leaf; #1825).
    """
    from discopt._flat_index import resolve_scalar_slot
    from discopt._relax.scalarize import scalar_leaf_value
    from discopt.modeling.core import (
        BinaryOp,
        Constant,
        IndexExpression,
        Parameter,
        SumExpression,
        SumOverExpression,
        UnaryOp,
    )

    def _extract(e) -> tuple[dict, float] | None:
        if isinstance(e, (Constant, Parameter)):
            # #1825: an array-valued leaf makes the body several rows, not one.
            # Summing it (the old ``float(np.sum(value))``) turned
            # ``x >= [5, 1]`` into ``x >= 6`` and mbigm certified a wrong optimum.
            value = scalar_leaf_value(e)
            return None if value is None else ({}, value)
        if isinstance(e, (Variable, IndexExpression)):
            # #941's single source of truth, which this layer was open-coding.
            # The hand-rolled version did ``int(e.index)``, which raises
            # TypeError on the tuple index of a 2-D variable -- crashing the
            # whole reformulation instead of declining one row -- and took a
            # negative index literally, returning an in-range slot belonging to
            # a *different* variable. ``None`` means "not one scalar slot" and
            # every caller below already falls back conservatively on it.
            slot = resolve_scalar_slot(e, model)
            return None if slot is None else ({slot: 1.0}, 0.0)
        if isinstance(e, UnaryOp) and e.op == "neg":
            inner = _extract(e.operand)
            if inner is None:
                return None
            neg_coeffs, neg_off = inner
            return {k: -v for k, v in neg_coeffs.items()}, -neg_off
        if isinstance(e, BinaryOp):
            if e.op == "+":
                le, ri = _extract(e.left), _extract(e.right)
                if le is None or ri is None:
                    return None
                lc, lo = le
                rc, ro = ri
                merged = dict(lc)
                for k, v in rc.items():
                    merged[k] = merged.get(k, 0.0) + v
                return merged, lo + ro
            if e.op == "-":
                le, ri = _extract(e.left), _extract(e.right)
                if le is None or ri is None:
                    return None
                lc, lo = le
                rc, ro = ri
                merged = dict(lc)
                for k, v in rc.items():
                    merged[k] = merged.get(k, 0.0) - v
                return merged, lo - ro
            if e.op == "*":
                le, ri = _extract(e.left), _extract(e.right)
                if le is not None and ri is not None:
                    lc, lo = le
                    rc, ro = ri
                    # Constant * linear or linear * constant
                    if not lc and not rc:
                        return {}, lo * ro
                    if not lc:
                        return {k: lo * v for k, v in rc.items()}, lo * ro
                    if not rc:
                        return {k: ro * v for k, v in lc.items()}, lo * ro
                return None
        if isinstance(e, SumOverExpression):
            # #1039: fold the indexed summation term by term. This is what
            # lets the OA/LOA row extraction see the ``exactly-one`` row that
            # ``make_disjunct``/``add_disjunction`` emits; returning None here
            # dropped that row from the master MILP, which was then free to
            # propose "no disjunct active" — a fixed-integer NLP that is
            # infeasible, so LOA reported ``status="unknown"`` on a trivially
            # feasible model. Succeeding here is safe for every caller because
            # the returned ``(c_vec, offset)`` IS the linearity witness.
            acc: dict = {}
            off_total = 0.0
            for t in e.terms:
                sub = _extract(t)
                if sub is None:
                    return None
                sub_coeffs, sub_off = sub
                for k, v in sub_coeffs.items():
                    acc[k] = acc.get(k, 0.0) + v
                off_total += sub_off
            return acc, off_total
        if isinstance(e, SumExpression):
            # Reduction over an array-valued operand: not a single row.
            return None
        return None

    result = _extract(expr)
    if result is None:
        return None

    coeffs_dict, offset = result
    c_vec = np.zeros(n_vars)
    for idx, val in coeffs_dict.items():
        if idx < n_vars:
            c_vec[idx] = val
    return c_vec, offset


def _precompute_lp_relaxation(model: Model) -> tuple | None:
    """Precompute LP relaxation data from the model for MBM.

    Returns (A_ub, b_ub, A_eq, b_eq, n_vars) or None if no linear
    constraints found or HiGHS is unavailable.
    """
    try:
        from discopt._relax.obbt import _extract_linear_constraints

        return _extract_linear_constraints(model)
    except (ImportError, Exception):
        return None


def _sumover_enabled() -> bool:
    """Is #1154 ``SumOverExpression`` support on? (``DISCOPT_GDP_SUMOVER``, default ON).

    Read only when a :class:`SumOverExpression` node is actually reached, so the
    flag costs nothing on any expression that does not contain one, and the ``=0``
    opt-out is byte-identical to the pre-#1154 walkers. Graduated default-ON by the
    §5 panel in ``docs/dev/issue-1154-gdp-sumover-panel-2026-09-04.md``.
    """
    from discopt.solver_tuning import current as _tuning

    return bool(_tuning().gdp_sumover)


def _sumover_terms(expr: Expression) -> list[Expression] | None:
    """``expr.terms`` when *expr* is an enabled ``SumOverExpression``, else ``None``.

    ``Σ[t1, …, tn]`` is treated by every walker in this module **exactly** as the
    left-folded chain ``t1 + … + tn`` — the desugaring the modeling layer could
    equally have produced. That equivalence is the whole contract; it is
    machine-checked per walker in ``python/tests/test_1154_gdp_sumover_hull.py``.
    """
    if not isinstance(expr, SumOverExpression):
        return None
    if not _sumover_enabled():
        return None
    return list(expr.terms)


class _BoundMemo:
    """#1565: per-call memo for :func:`_bound_expression` and :func:`_bound_error`.

    Both walks are pure functions of a node and the (unchanging, for the duration
    of one call) variable bounds, so a node's interval and endpoint errors are
    computed once and reused at every other parent that shares it. Without this,
    a DAG with shared subexpressions -- an OMLT reduced-space network, where each
    layer's activations feed every unit of the next layer -- is walked as its
    expanded *tree*, and ``_bound_error`` re-walks each node's subtree with
    ``_bound_expression`` several times on top. Measured on OMLT 2-input
    sigmoid nets: 15.0 s for one ``bound_expression_error`` on a 2,889-node DAG
    (148,206 tree nodes), and ``solve(time_limit=30)`` running 65 s on 25x3.

    Keyed by ``id(node)`` **with the node held alongside the value**, so no key
    can be recycled by a garbage-collected node while the memo is alive (the
    #1283/#1555 ``id()``-aliasing hazard). A memo must not outlive a change to any
    variable bound it read; every caller scopes one to a single pass over a fixed
    box.
    """

    __slots__ = ("bounds", "errors")

    def __init__(self) -> None:
        self.bounds: dict[int, tuple[Expression, tuple[float, float]]] = {}
        self.errors: dict[int, tuple[Expression, tuple[float, float]]] = {}

    def put_bound(self, expr: Expression, res: tuple[float, float]) -> tuple[float, float]:
        self.bounds[id(expr)] = (expr, res)
        return res

    def put_error(self, expr: Expression, res: tuple[float, float]) -> tuple[float, float]:
        self.errors[id(expr)] = (expr, res)
        return res


def _bound_expression(
    expr: Expression,
    model: Model,
    memo: _BoundMemo | None = None,
) -> tuple[float, float]:
    """Compute interval bounds [lo, hi] for an expression via interval arithmetic.

    Traverses the expression DAG and propagates bounds from variable
    bounds through operations. Each distinct node is evaluated once per *memo*
    (a fresh one per call when omitted), so the cost is linear in DAG size
    (#1565). Pass a shared :class:`_BoundMemo` only across calls over the same
    variable box.

    Returns
    -------
    tuple of (float, float)
        (lower_bound, upper_bound) of the expression.
    """
    # The memo check and the ``memo.put_bound`` at every return live in this one
    # function, not a wrapper, so the recursion stays one Python frame per
    # expression level (a deep left-folded sum must not hit the recursion limit
    # at half the depth it used to).
    if memo is None:
        memo = _BoundMemo()
    hit = memo.bounds.get(id(expr))
    if hit is not None:
        return hit[1]
    if isinstance(expr, Variable):
        lo = float(np.min(expr.lb))
        hi = float(np.max(expr.ub))
        return memo.put_bound(expr, (lo, hi))

    if isinstance(expr, Constant):
        val = float(np.min(expr.value))
        val_max = float(np.max(expr.value))
        return memo.put_bound(expr, (val, val_max))

    if isinstance(expr, IndexExpression):
        base_lo, base_hi = _bound_expression(expr.base, model, memo)
        # For indexed expressions on variables, get tighter bounds
        if isinstance(expr.base, Variable):
            v = expr.base
            idx = expr.index
            lb_slice = v.lb[idx] if v.shape != () else v.lb
            ub_slice = v.ub[idx] if v.shape != () else v.ub
            lo = float(np.min(lb_slice))
            hi = float(np.max(ub_slice))
            return memo.put_bound(expr, (lo, hi))
        return memo.put_bound(expr, (base_lo, base_hi))

    terms = _sumover_terms(expr)
    if terms is not None:
        # Left-fold, exactly as the equivalent ``t1 + ... + tn`` BinaryOp chain
        # is folded by the ``BinaryOp`` "+" branch below. Infinities are handled
        # explicitly
        # rather than left to ``+``: an unbounded-below term and an unbounded-above
        # term in the same sum would make plain float addition produce NaN,
        # which reads as "no bound" nowhere and silently poisons the big-M.
        lo_total = 0.0
        hi_total = 0.0
        for term in terms:
            t_lo, t_hi = _bound_expression(term, model, memo)
            lo_total = -np.inf if (lo_total == -np.inf or t_lo == -np.inf) else lo_total + t_lo
            hi_total = np.inf if (hi_total == np.inf or t_hi == np.inf) else hi_total + t_hi
        return memo.put_bound(expr, (float(lo_total), float(hi_total)))

    if isinstance(expr, BinaryOp):
        left_lo, left_hi = _bound_expression(expr.left, model, memo)
        right_lo, right_hi = _bound_expression(expr.right, model, memo)

        if expr.op == "+":
            return memo.put_bound(expr, (left_lo + right_lo, left_hi + right_hi))
        elif expr.op == "-":
            return memo.put_bound(expr, (left_lo - right_hi, left_hi - right_lo))
        elif expr.op == "*":
            products = [
                left_lo * right_lo,
                left_lo * right_hi,
                left_hi * right_lo,
                left_hi * right_hi,
            ]
            return memo.put_bound(expr, (min(products), max(products)))
        elif expr.op == "/":
            if right_lo > 0 or right_hi < 0:
                # Divisor doesn't cross zero
                quotients = [
                    left_lo / right_lo,
                    left_lo / right_hi,
                    left_hi / right_lo,
                    left_hi / right_hi,
                ]
                return memo.put_bound(expr, (min(quotients), max(quotients)))
            return memo.put_bound(expr, (-np.inf, np.inf))
        elif expr.op == "**":
            if isinstance(expr.right, Constant):
                p = float(expr.right.value)
                if p == int(p) and p > 0:
                    p_int = int(p)
                    # #1520: float64 power, so an overflow is a signed inf (a
                    # valid, looser endpoint) rather than Python's OverflowError.
                    with np.errstate(over="ignore"):
                        vals = [
                            float(np.float64(left_lo) ** p_int),
                            float(np.float64(left_hi) ** p_int),
                        ]
                    if p_int % 2 == 0 and left_lo < 0 < left_hi:
                        # Even power over a base that straddles 0: x^p is not
                        # monotone here — its interior minimum is at x=0
                        # (value 0), which the endpoint evaluations miss.
                        # The rigorous range is [0, max(lb^p, ub^p)]
                        # (C-34/FR-1). This generalizes the p==2 case that
                        # previously handled only the quadratic straddle;
                        # p>=4 formerly fell to the endpoint-only path below
                        # and under-approximated the range → invalid box →
                        # false certified optimum.
                        return memo.put_bound(expr, (0.0, max(vals)))
                    # Odd power (monotone on any interval) or one-signed even
                    # power (monotone on that interval): the range is spanned
                    # by the endpoints.
                    return memo.put_bound(expr, (min(vals), max(vals)))
                frac = _fractional_power_bounds(left_lo, left_hi, p)
                if frac is not None:
                    return memo.put_bound(expr, frac)
            return memo.put_bound(expr, (-np.inf, np.inf))

    if isinstance(expr, UnaryOp):
        arg_lo, arg_hi = _bound_expression(expr.operand, model, memo)
        if expr.op == "neg":
            return memo.put_bound(expr, (-arg_hi, -arg_lo))
        elif expr.op == "abs":
            vals = [abs(arg_lo), abs(arg_hi)]
            if arg_lo <= 0 <= arg_hi:
                return memo.put_bound(expr, (0.0, max(vals)))
            return memo.put_bound(expr, (min(vals), max(vals)))
        return memo.put_bound(expr, (-np.inf, np.inf))

    if isinstance(expr, FunctionCall):
        arg_lo, arg_hi = _bound_expression(expr.args[0], model, memo)
        if expr.func_name == "exp":
            lo = np.exp(arg_lo) if np.isfinite(arg_lo) else 0.0
            hi = np.exp(arg_hi) if np.isfinite(arg_hi) else np.inf
            return memo.put_bound(expr, (lo, hi))
        elif expr.func_name == "log":
            lo = np.log(max(arg_lo, 1e-300)) if arg_lo > 0 else -np.inf
            hi = np.log(max(arg_hi, 1e-300)) if arg_hi > 0 else -np.inf
            return memo.put_bound(expr, (lo, hi))
        elif expr.func_name == "abs":
            vals = [abs(arg_lo), abs(arg_hi)]
            if arg_lo <= 0 <= arg_hi:
                return memo.put_bound(expr, (0.0, max(vals)))
            return memo.put_bound(expr, (min(vals), max(vals)))
        elif expr.func_name == "sqrt":
            lo = np.sqrt(max(arg_lo, 0.0))
            hi = np.sqrt(max(arg_hi, 0.0)) if np.isfinite(arg_hi) else np.inf
            return memo.put_bound(expr, (lo, hi))
        elif expr.func_name in ("sin", "cos"):
            # Conservative for trig
            return memo.put_bound(expr, (-1.0, 1.0))
        elif expr.func_name == "neg":
            return memo.put_bound(expr, (-arg_hi, -arg_lo))

    # Fallback: unknown expression type
    return memo.put_bound(expr, (-np.inf, np.inf))


def _fractional_power_bounds(lo: float, hi: float, p: float) -> tuple[float, float] | None:
    """#1767: enclosure of ``x**p`` for a constant non-integer exponent *p* over a
    base enclosure ``[lo, hi]``, or ``None`` to decline.

    A real power with a non-integer exponent is defined only for ``x >= 0``, so a
    base enclosure that reaches below 0 is declined (the caller answers
    ``(-inf, inf)`` and the big-M refuses loudly). On ``[lo, hi]`` with ``lo >= 0``
    the function is monotone -- increasing for ``p > 0``, decreasing for ``p < 0``
    -- so the range is spanned by the endpoint powers. ``0**p`` for ``p < 0`` and
    ``inf**p`` for ``p > 0`` evaluate to ``inf``, so those endpoints stay unbounded.
    Integer exponents are not handled here (the ``**`` branch owns them).
    """
    if not np.isfinite(p) or p == int(p):
        return None
    if not (lo >= 0.0 and lo <= hi) or np.isnan(hi):
        return None
    with np.errstate(over="ignore", divide="ignore"):
        a = float(np.float64(lo) ** p)
        b = float(np.float64(hi) ** p)
    return (min(a, b), max(a, b))


def _fractional_power_slope(v: float, err: float, p: float) -> float:
    """Largest ``|d/dx x**p| = |p| x**(p-1)`` over ``[v - err, v + err]``.

    The slope is monotone in ``x`` on ``x > 0``, so its maximum is at one end of the
    ball. If the ball reaches ``x <= 0`` (where the slope is unbounded for
    ``p < 1`` and the power undefined below 0) or ``v`` is infinite, the slope is
    ``inf``; :func:`_mul0` still turns a zero *err* into zero propagated error.
    """
    if err == 0.0:
        return 0.0
    left = v - err
    if not (np.isfinite(v) and left > 0.0):
        return np.inf
    with np.errstate(over="ignore"):
        s = abs(p) * max(np.float64(left) ** (p - 1.0), np.float64(v + err) ** (p - 1.0))
    return float(s)


def bound_expression_error(
    expr: Expression, model: Model, memo: _BoundMemo | None = None
) -> tuple[float, float]:
    """#1397: upper bounds on the floating-point error in each of
    :func:`_bound_expression`'s two endpoints for *expr*, as ``(err_lo, err_hi)``.

    :func:`_bound_expression` is plain ``float`` interval arithmetic with **no
    outward rounding**, so its ``lo`` is not a rigorous under-estimate of the true
    infimum and its ``hi`` is not a rigorous over-estimate of the supremum. That is
    harmless where the interval is only used to *size* something (a big-M, a
    heuristic width) but not where a caller reads a *sign* off it: comparing ``lo``
    against an absolute margin such as ``factorable_reform._ZERO_MARGIN`` (1e-9)
    silently assumes the endpoint is exact, and one ulp of a 1e17 sum is 16.0 --
    ten orders of magnitude past the margin. The true infimum is no less than
    ``lo - err_lo`` and the true supremum no more than ``hi + err_hi``.

    This walks the same DAG with the same operator dispatch and propagates a
    first-order bound: each node contributes a few ulps of its own endpoint, plus
    its operands' errors scaled by the node's local sensitivity -- ``|b|`` and
    ``|a|`` for a product ``a*b``, ``1/|b|`` and ``|a|/b^2`` for ``a/b``,
    ``p|v|^(p-1)`` for an integer power (the largest slope over the base's error
    ball for a non-integer power on a nonnegative base, #1767), ``|f'|`` for a
    monotone unary function.
    Because ``_bound_expression`` picks each endpoint from a specific operand
    corner, the error is tracked **per endpoint** rather than as one scalar: a
    denominator ``0.01 + x`` with ``x.ub = +inf`` has an exact lower endpoint and an
    infinite upper one, and collapsing the two would refuse a perfectly sound
    sign inference. (Measured: doing so lost 92 of 303 denominator clears on the
    in-repo corpus, three ``heatexch_gen*`` instances going to zero.)

    A node whose sensitivity this cannot bound -- an unknown call, a non-integer
    power on a base that can be negative, a non-sign-definite quotient -- yields
    ``inf`` for the affected endpoint, which makes a caller's sign test on that
    endpoint fail: the inference is refused rather than guessed at (CLAUDE.md §3).

    The bound is deliberately loose. Every caller uses it to *widen* a refusal
    threshold, so an over-estimate costs only the strength of an optional rewrite
    while an under-estimate costs soundness.

    Linear in DAG size: each node's error and interval are computed once per
    *memo* (#1565). A caller making several calls over one variable box may pass
    one shared :class:`_BoundMemo` to reuse work across them.
    """
    return _bound_error(expr, model, memo)


#: ``ROUNDOFF_OPS`` ulps per node: ``_bound_expression`` evaluates up to four
#: products/quotients and a min/max at one node, so a single ulp under-counts.
#: Same constant as :func:`~discopt._relax._numeric.roundoff_slack`.
_ERR_UNIT = ROUNDOFF_OPS * FLOAT_EPS


def _mul0(a: float, b: float) -> float:
    """``a * b`` with ``0 * inf == 0``: a zero sensitivity transmits no error, and
    plain float multiplication would make it NaN (which compares False everywhere
    and would silently *pass* a sign test)."""
    if a == 0.0 or b == 0.0:
        return 0.0
    return a * b


def _corner_err(val: float, err: float) -> float:
    """Local round-off of an endpoint plus the propagated error reaching it."""
    return err + _ERR_UNIT * abs(val)


def _pick_err(target: float, corners) -> float:
    """Error of whichever corner value ``_bound_expression`` selected for *target*.

    ``max`` over ties, and over *all* corners when float comparison matches none of
    them (an ``inf`` endpoint), so an unmatched endpoint is never given a smaller
    error than the corners it was chosen from.
    """
    matched = [e for v, e in corners if v == target]
    return max(matched) if matched else max((e for _, e in corners), default=np.inf)


def _bound_error(
    expr: Expression, model: Model, memo: _BoundMemo | None = None
) -> tuple[float, float]:
    # Operand errors and every interval come from *memo*, so no subtree is
    # re-walked (#1565); one frame per level, as in :func:`_bound_expression`.
    if memo is None:
        memo = _BoundMemo()
    hit = memo.errors.get(id(expr))
    if hit is not None:
        return hit[1]
    if isinstance(expr, (Variable, Constant, IndexExpression)):
        # Declared bounds and literals are exact floats: no arithmetic happened.
        return memo.put_error(expr, (0.0, 0.0))

    terms = _sumover_terms(expr)
    if terms is not None:
        # Left-fold of ``t1 + ... + tn``: lo endpoints add, hi endpoints add.
        e_lo = e_hi = 0.0
        for term in terms:
            t_elo, t_ehi = _bound_error(term, model, memo)
            e_lo += t_elo
            e_hi += t_ehi
        lo, hi = _bound_expression(expr, model, memo)
        return memo.put_error(expr, (_corner_err(lo, e_lo), _corner_err(hi, e_hi)))

    if isinstance(expr, BinaryOp):
        el_lo, el_hi = _bound_error(expr.left, model, memo)
        er_lo, er_hi = _bound_error(expr.right, model, memo)
        l_lo, l_hi = _bound_expression(expr.left, model, memo)
        r_lo, r_hi = _bound_expression(expr.right, model, memo)
        lo, hi = _bound_expression(expr, model, memo)

        if expr.op == "+":
            return memo.put_error(
                expr, (_corner_err(lo, el_lo + er_lo), _corner_err(hi, el_hi + er_hi))
            )
        if expr.op == "-":
            # lo = l_lo - r_hi, hi = l_hi - r_lo.
            return memo.put_error(
                expr, (_corner_err(lo, el_lo + er_hi), _corner_err(hi, el_hi + er_lo))
            )
        if expr.op in ("*", "/"):
            # ``_bound_expression`` takes min/max over the four corner combinations,
            # so each endpoint inherits the error of whichever corner attained it.
            corners: list[tuple[float, float]] = []
            for a, ea in ((l_lo, el_lo), (l_hi, el_hi)):
                for b, eb in ((r_lo, er_lo), (r_hi, er_hi)):
                    if expr.op == "*":
                        corners.append((a * b, _mul0(abs(b), ea) + _mul0(abs(a), eb)))
                    else:
                        if not (r_lo > 0 or r_hi < 0):
                            return memo.put_error(
                                expr, (np.inf, np.inf)
                            )  # matches the (-inf, inf) branch
                        ab = abs(b)
                        corners.append((a / b, ea / ab + _mul0(abs(a), eb) / (ab * ab)))
            return memo.put_error(
                expr,
                (
                    (
                        _corner_err(lo, _pick_err(lo, corners)),
                        _corner_err(hi, _pick_err(hi, corners)),
                    )
                ),
            )
        if expr.op == "**" and isinstance(expr.right, Constant):
            p = float(expr.right.value)
            if p == int(p) and p > 0:
                p_int = int(p)
                if p_int % 2 == 0 and l_lo < 0 < l_hi:
                    # ``(0.0, max(lb**p, ub**p))``: the lower endpoint is the exact
                    # literal 0.0, so only the upper one carries error.
                    e_up = max(
                        _mul0(p_int * abs(l_lo) ** (p_int - 1), el_lo),
                        _mul0(p_int * abs(l_hi) ** (p_int - 1), el_hi),
                    )
                    return memo.put_error(expr, (0.0, _corner_err(hi, e_up)))
                vals = (
                    (l_lo**p_int, _mul0(p_int * abs(l_lo) ** (p_int - 1), el_lo)),
                    (l_hi**p_int, _mul0(p_int * abs(l_hi) ** (p_int - 1), el_hi)),
                )
                return memo.put_error(
                    expr,
                    (
                        (
                            _corner_err(lo, _pick_err(lo, vals)),
                            _corner_err(hi, _pick_err(hi, vals)),
                        )
                    ),
                )
            if _fractional_power_bounds(l_lo, l_hi, p) is not None:
                # #1767: ``x**p`` is monotone on the nonnegative base enclosure, so
                # each result endpoint is the power of one base endpoint. Its error
                # is that base endpoint's error times the *largest* slope over the
                # base's error ball (mean value theorem), not the slope at the
                # endpoint: for 0 < p < 1 the slope grows toward 0, so the
                # first-order slope at the endpoint would under-count.
                with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                    vals = (
                        (
                            float(np.float64(l_lo) ** p),
                            _mul0(_fractional_power_slope(l_lo, el_lo, p), el_lo),
                        ),
                        (
                            float(np.float64(l_hi) ** p),
                            _mul0(_fractional_power_slope(l_hi, el_hi, p), el_hi),
                        ),
                    )
                return memo.put_error(
                    expr,
                    (
                        _corner_err(lo, _pick_err(lo, vals)),
                        _corner_err(hi, _pick_err(hi, vals)),
                    ),
                )
        return memo.put_error(expr, (np.inf, np.inf))

    if isinstance(expr, UnaryOp):
        e_lo, e_hi = _bound_error(expr.operand, model, memo)
        if expr.op == "neg":
            return memo.put_error(expr, (e_hi, e_lo))  # lo = -arg_hi, hi = -arg_lo
        if expr.op == "abs":
            return memo.put_error(expr, (_abs_error(expr.operand, model, e_lo, e_hi, memo)))
        return memo.put_error(expr, (np.inf, np.inf))

    if isinstance(expr, FunctionCall):
        e_lo, e_hi = _bound_error(expr.args[0], model, memo)
        a_lo, a_hi = _bound_expression(expr.args[0], model, memo)
        lo, hi = _bound_expression(expr, model, memo)
        name = expr.func_name
        if name == "neg":
            return memo.put_error(expr, (e_hi, e_lo))
        if name == "abs":
            return memo.put_error(expr, (_abs_error(expr.args[0], model, e_lo, e_hi, memo)))
        if name in ("sin", "cos"):
            # ``_bound_expression`` returns the exact literal interval (-1, 1)
            # regardless of the argument, so no argument error reaches it.
            return memo.put_error(expr, (0.0, 0.0))
        if name in ("exp", "log", "sqrt"):
            # All three are monotone increasing, so endpoints map to endpoints and
            # the local sensitivity is |f'| at that endpoint.
            d_lo, d_hi = _MONOTONE_SLOPE[name](a_lo, a_hi)
            return memo.put_error(
                expr, (_corner_err(lo, _mul0(d_lo, e_lo)), _corner_err(hi, _mul0(d_hi, e_hi)))
            )
        return memo.put_error(expr, (np.inf, np.inf))

    return memo.put_error(expr, (np.inf, np.inf))


def _abs_error(
    operand: Expression, model: Model, e_lo: float, e_hi: float, memo: _BoundMemo
) -> tuple[float, float]:
    """``|u|``: on a straddling interval the lower endpoint is the exact literal
    0.0; otherwise both endpoints are ``|`` of an operand endpoint, so the error
    can arrive from either side."""
    a_lo, a_hi = _bound_expression(operand, model, memo)
    worst = max(e_lo, e_hi)
    if a_lo <= 0 <= a_hi:
        return 0.0, worst
    return worst, worst


def _slope_exp(a_lo: float, a_hi: float) -> tuple[float, float]:
    # f' = exp, evaluated at each endpoint; matches the branch's own clamping.
    lo = float(np.exp(a_lo)) if np.isfinite(a_lo) else 0.0
    hi = float(np.exp(a_hi)) if np.isfinite(a_hi) else np.inf
    return lo, hi


def _slope_log(a_lo: float, a_hi: float) -> tuple[float, float]:
    # f' = 1/x on x > 0; the branch returns -inf where the endpoint is <= 0, and
    # an infinite endpoint carries no usable sensitivity.
    lo = 1.0 / a_lo if a_lo > 0 else np.inf
    hi = 1.0 / a_hi if a_hi > 0 else np.inf
    return lo, hi


def _slope_sqrt(a_lo: float, a_hi: float) -> tuple[float, float]:
    # f' = 1/(2*sqrt(x)); the branch clamps a negative endpoint to 0, where the
    # slope is unbounded.
    lo = 1.0 / (2.0 * np.sqrt(a_lo)) if a_lo > 0 else np.inf
    hi = 1.0 / (2.0 * np.sqrt(a_hi)) if a_hi > 0 else np.inf
    return float(lo), float(hi)


_MONOTONE_SLOPE = {"exp": _slope_exp, "log": _slope_log, "sqrt": _slope_sqrt}


# ── Indicator constraint reformulation ──


def _reformulate_indicator(
    ic: _IndicatorConstraint,
    model: Model,
    lp_data: tuple | None = None,
) -> list[Constraint]:
    """Reformulate an indicator constraint to big-M constraints.

    ``if indicator == active_value then constraint`` becomes:
    - For ``body <= 0``: ``body <= M * (1 - indicator)``
      i.e. ``body - M*(1-indicator) <= 0``
    - For ``body >= 0``: ``body >= -M * (1 - indicator)``
      i.e. ``body + M*(1-indicator) >= 0``
    - For ``body == 0``: both ``<=`` and ``>=`` reformulations

    When ``active_value == 0``, the logic flips: we use ``M * indicator``
    instead of ``M * (1 - indicator)``.
    """
    con = ic.constraint
    y = ic.indicator
    if lp_data is not None:
        M = _compute_big_m_lp(con, model, lp_data)
    else:
        M = _compute_big_m(con, model)

    if ic.active_value == 1:
        # When y=1 constraint is active; when y=0, relaxed by M
        deactivation_expr = _wrap(M) * (_wrap(1.0) - y)
    else:
        # When y=0 constraint is active; when y=1, relaxed by M
        deactivation_expr = _wrap(M) * y

    result = []

    if con.sense == "<=":
        # body <= M*(1-y) => body - M*(1-y) <= 0
        new_body = con.body - deactivation_expr
        result.append(Constraint(body=new_body, sense="<=", rhs=0.0, name=con.name))

    elif con.sense == ">=":
        # body >= -M*(1-y) => body + M*(1-y) >= 0
        new_body = con.body + deactivation_expr
        result.append(Constraint(body=new_body, sense=">=", rhs=0.0, name=con.name))

    elif con.sense == "==":
        # body == 0 when active => -M*(1-y) <= body <= M*(1-y)
        name_le = f"{con.name}_le" if con.name else None
        name_ge = f"{con.name}_ge" if con.name else None
        result.append(
            Constraint(
                body=con.body - deactivation_expr,
                sense="<=",
                rhs=0.0,
                name=name_le,
            )
        )
        result.append(
            Constraint(
                body=con.body + deactivation_expr,
                sense=">=",
                rhs=0.0,
                name=name_ge,
            )
        )

    return result


# ── Disjunctive constraint reformulation ──


def _reformulate_disjunction(
    dc: _DisjunctiveConstraint,
    model: Model,
    add_aux_binary,
    lp_data: tuple | None = None,
    selectors: Optional[list[Variable]] = None,
) -> tuple[list[Variable], list[Constraint]]:
    """Reformulate a disjunction via big-M.

    For ``either_or([[g1<=0, g2<=0], [h1<=0, h2<=0]])``:
    1. Introduce binary selectors y_0, y_1 with y_0 + y_1 == 1
    2. For each disjunct k and constraint j in disjunct k:
       g_j(x) <= M_j * (1 - y_k)

    ``selectors`` (#1775 G1.1) are the disjunction's own indicator columns, read
    by the user through ``_DisjunctiveConstraint.indicators``. They are used in
    place of generated selectors, and step 1 is skipped: the model already
    carries their ``exactly(1)`` row.

    Under OR semantics (#1775 G1.4) step 1 is ``sum_k y_k >= 1``.

    Returns new variables and constraints.
    """
    _require_supported_semantics(dc, _BIG_M_SEMANTICS)
    n_disjuncts = len(dc.disjuncts)
    new_vars = []
    new_cons = []

    if selectors is not None:
        if len(selectors) != n_disjuncts:
            raise ValueError(
                f"big-M of disjunction {dc.name or 'anon'!r}: {len(selectors)} selectors "
                f"for {n_disjuncts} disjuncts."
            )
        selectors = list(selectors)
    else:
        # Create selector binaries
        selectors = []
        for k in range(n_disjuncts):
            y_k = add_aux_binary(f"disj_{dc.name or 'anon'}_{k}")
            selectors.append(y_k)
            new_vars.append(y_k)

        # Sum of selectors == 1 (exactly one disjunct active)
        if n_disjuncts == 2:
            sum_expr = selectors[0] + selectors[1]
        else:
            sum_expr = selectors[0]
            for k in range(1, n_disjuncts):
                sum_expr = sum_expr + selectors[k]

        new_cons.append(
            Constraint(
                body=sum_expr - _wrap(1.0),
                sense=">=" if _is_or(dc) else "==",
                rhs=0.0,
                name=f"_gdp_select_{dc.name}" if dc.name else None,
            )
        )

    # Big-M reformulation for each constraint in each disjunct
    for k, disjunct in enumerate(dc.disjuncts):
        y_k = selectors[k]
        for j, item in enumerate(disjunct):
            # Handle nested disjunctions
            if isinstance(item, _DisjunctiveConstraint):
                inner_vars, inner_cons = _reformulate_disjunction_nested(
                    item, model, add_aux_binary, parent_selector=y_k
                )
                new_vars.extend(inner_vars)
                new_cons.extend(inner_cons)
                continue

            con = item
            if lp_data is not None:
                M = _compute_big_m_lp(con, model, lp_data)
            else:
                M = _compute_big_m(con, model)
            deactivation = _wrap(M) * (_wrap(1.0) - y_k)

            if con.sense == "<=":
                new_body = con.body - deactivation
                new_cons.append(
                    Constraint(
                        body=new_body,
                        sense="<=",
                        rhs=0.0,
                        name=f"_gdp_{dc.name}_d{k}_c{j}" if dc.name else None,
                    )
                )
            elif con.sense == ">=":
                new_body = con.body + deactivation
                new_cons.append(
                    Constraint(
                        body=new_body,
                        sense=">=",
                        rhs=0.0,
                        name=f"_gdp_{dc.name}_d{k}_c{j}" if dc.name else None,
                    )
                )
            elif con.sense == "==":
                new_cons.append(
                    Constraint(
                        body=con.body - deactivation,
                        sense="<=",
                        rhs=0.0,
                        name=f"_gdp_{dc.name}_d{k}_c{j}_le" if dc.name else None,
                    )
                )
                new_cons.append(
                    Constraint(
                        body=con.body + deactivation,
                        sense=">=",
                        rhs=0.0,
                        name=f"_gdp_{dc.name}_d{k}_c{j}_ge" if dc.name else None,
                    )
                )

    return new_vars, new_cons


def _reformulate_disjunction_simplex(
    dc: _DisjunctiveConstraint,
    model: Model,
    add_weight,
    index: int,
) -> tuple[list[Variable], list[Constraint]]:
    """Reformulate a disjunction with the exact continuous lowering of #1182.

    Thin adapter over :func:`discopt._relax.simplex_lowering.
    lower_disjunction_simplex`: the mathematics, the refusals and the size
    accounting live there. The record is appended to ``model._simplex_lowerings``
    so the four size quantities of #1182 requirement 4 stay inspectable after
    the pass, and so a caller can see which disjunctions became witnesses rather
    than selectors.

    ``new_vars`` is returned empty on purpose: the weight variables are already
    registered on ``model`` by ``add_weight`` (as ``_add_aux_binary`` does for
    selectors), and the caller uses the returned list only for logging.
    """
    from discopt._relax.simplex_lowering import (
        SimplexLoweringRefused,
        lower_disjunction_simplex,
    )

    # #1779: Theorem 1's row ``sum_j lam_j p_j(x) <= 0`` keeps every predicate in
    # the model at every point, so a predicate undefined on part of the box
    # imposes its domain on the other disjuncts exactly as a big-M row does.
    for disjunct in dc.disjuncts:
        for item in disjunct:
            if isinstance(item, Constraint):
                gap = _domain_gap(item.body, model)
                if gap is not None:
                    err = _domain_gap_error(item, gap, "the simplex lowering")
                    raise SimplexLoweringRefused(str(err))

    record, rows = lower_disjunction_simplex(dc, add_weight, index=index)
    model._simplex_lowerings.append(record)
    return [], rows


def _reformulate_disjunction_nested(
    dc: _DisjunctiveConstraint,
    model: Model,
    add_aux_binary,
    parent_selector,
) -> tuple[list[Variable], list[Constraint]]:
    """Reformulate a nested disjunction linked to a parent selector.

    When parent_selector = 0, all inner selectors must be 0.
    When parent_selector = 1, exactly one inner selector must be 1 -- or, under
    OR semantics (#1775 G1.4), at least one: ``sum_k y_k >= parent`` together
    with ``y_k <= parent`` for each ``k``.
    """
    _require_supported_semantics(dc, _BIG_M_SEMANTICS)
    n_disjuncts = len(dc.disjuncts)
    new_vars = []
    new_cons = []

    # Create inner selector binaries
    selectors = []
    for k in range(n_disjuncts):
        y_k = add_aux_binary(f"nested_{dc.name or 'anon'}_{k}")
        selectors.append(y_k)
        new_vars.append(y_k)

    # sum(inner selectors) == parent_selector
    sum_expr = selectors[0]
    for k in range(1, n_disjuncts):
        sum_expr = sum_expr + selectors[k]

    new_cons.append(
        Constraint(
            body=sum_expr - parent_selector,
            sense=">=" if _is_or(dc) else "==",
            rhs=0.0,
            name=f"_gdp_nested_select_{dc.name}" if dc.name else None,
        )
    )
    if _is_or(dc):
        new_cons.extend(
            Constraint(body=y_k - parent_selector, sense="<=", rhs=0.0) for y_k in selectors
        )

    # Big-M for each inner constraint
    for k, disjunct in enumerate(dc.disjuncts):
        y_k = selectors[k]
        for j, item in enumerate(disjunct):
            if isinstance(item, _DisjunctiveConstraint):
                # Recursive nesting
                inner_vars, inner_cons = _reformulate_disjunction_nested(
                    item, model, add_aux_binary, parent_selector=y_k
                )
                new_vars.extend(inner_vars)
                new_cons.extend(inner_cons)
                continue

            con = item
            M = _compute_big_m(con, model)
            deactivation = _wrap(M) * (_wrap(1.0) - y_k)

            if con.sense == "<=":
                new_cons.append(
                    Constraint(
                        body=con.body - deactivation,
                        sense="<=",
                        rhs=0.0,
                        name=f"_gdp_nested_{dc.name}_d{k}_c{j}" if dc.name else None,
                    )
                )
            elif con.sense == ">=":
                new_cons.append(
                    Constraint(
                        body=con.body + deactivation,
                        sense=">=",
                        rhs=0.0,
                        name=f"_gdp_nested_{dc.name}_d{k}_c{j}" if dc.name else None,
                    )
                )
            elif con.sense == "==":
                new_cons.append(
                    Constraint(
                        body=con.body - deactivation,
                        sense="<=",
                        rhs=0.0,
                        name=f"_gdp_nested_{dc.name}_d{k}_c{j}_le" if dc.name else None,
                    )
                )
                new_cons.append(
                    Constraint(
                        body=con.body + deactivation,
                        sense=">=",
                        rhs=0.0,
                        name=f"_gdp_nested_{dc.name}_d{k}_c{j}_ge" if dc.name else None,
                    )
                )

    return new_vars, new_cons


# ── Hull reformulation ──


def _reformulate_disjunction_hull(
    dc: _DisjunctiveConstraint,
    model: Model,
    add_aux_binary,
    eps: float = 1e-8,
    selectors: Optional[list[Variable]] = None,
    prefix: Optional[str] = None,
    child_selectors: Optional[dict[int, list[Variable]]] = None,
) -> tuple[list[Variable], list[Constraint]]:
    """Reformulate a disjunction via convex hull relaxation.

    For each original variable x_j appearing in the disjunction, creates
    disaggregated copies v_{j,k} for each disjunct k, with:

    - Selector binaries y_k with sum(y_k) == 1
    - Aggregation: x_j == sum_k(v_{j,k})
    - Bound linking: dlb_k * y_k <= v_{j,k} <= dub_k * y_k
    - Linear constraints: substitute x -> v_k, multiply RHS by y_k
    - Nonlinear constraints: perspective form with clamped y_k

    Parameters
    ----------
    dc : _DisjunctiveConstraint
        The disjunction to reformulate.
    model : Model
        Model with variable bound information.
    add_aux_binary : callable
        Factory for creating auxiliary binary variables.
    eps : float
        Small positive constant for clamping y_k in perspective functions.
    selectors : list of Variable, optional
        Existing binaries to use as ``y_k`` (one per disjunct) instead of new
        auxiliaries. The caller owns their select-one row, so none is emitted
        (#1615 C-10a: the indicators of a ``Model.add_disjunction`` block, whose
        ``exactly(1)`` row the model already carries).
    prefix : str, optional
        Name stem of the generated variables and rows (default: ``dc.name``). A
        nested disjunction gets one derived from its parent's, so two unnamed
        children cannot collide.
    child_selectors : dict, optional
        ``id`` of a nested disjunction -> the selectors it is hulled on, for a
        nested ``add_disjunction`` block (its indicators, whose ``sum == parent``
        row the model carries). Any other nested disjunction gets fresh selectors.

    Nesting (#1775 G2.1) is lowered from the leaves up, as Pyomo's ``gdp.hull``
    does: a disjunction nested in disjunct ``k`` is hulled first, on its own fresh
    selectors, and everything that emits -- its select-one row, its aggregation
    rows ``x = sum v``, its bound links and its hulled rows -- becomes rows of
    disjunct ``k``. The variables the child (and, recursively, its own children)
    introduced -- selectors, disaggregated copies, a nested block's indicators --
    are **local** to disjunct ``k``: they are not disaggregated again but, being
    zero whenever ``y_k`` is, are already disjunct ``k``'s own copies, and enter the
    parent's perspective as ``w / yhat_k``. Every variable the child shares with the
    outside is disaggregated by the parent as usual. Re-hulling then turns the
    child's ``sum y_c = 1`` into ``sum y_c = y_k`` and its ``x = sum v`` into
    ``x_k = sum v``: the exact nested hull at every integer point.

    A local needs no bound link of its own at the parent: each is tied to a selector
    at the level that created it (a copy by ``dlb*y_c <= v <= dub*y_c``), and each
    selector to ``y_k`` by its select row (the re-hulled ``sum y_c = y_k``, or a
    block's ``sum == parent`` row), so ``w <= max(dub, 0)*y_k`` and its lower twin
    are already implied. A local that also appears in a sibling disjunct (a nested
    block's indicator used in a sibling's rows) cannot be disjunct ``k``'s copy
    there, so it is treated as shared and disaggregated like any other variable --
    exact too: at ``y_k = 1`` the sibling copies are pinned to 0, and at ``y_k = 0``
    the variable itself is 0 (its select row), so the sibling sees its true value.

    Returns
    -------
    tuple of (list[Variable], list[Constraint])
        New variables and constraints.
    """
    _require_supported_semantics(dc)
    n_disjuncts = len(dc.disjuncts)
    new_vars: list[Variable] = []
    new_cons: list[Constraint] = []
    if prefix is None:
        prefix = dc.name or "anon"

    if selectors is not None:
        if len(selectors) != n_disjuncts:
            raise ValueError(
                f"hull of disjunction {prefix!r}: {len(selectors)} selectors for "
                f"{n_disjuncts} disjuncts."
            )
        selectors = list(selectors)
    else:
        # --- Selector binaries y_k with sum == 1 ---
        selectors = []
        for k in range(n_disjuncts):
            y_k = add_aux_binary(f"hull_{prefix}_{k}")
            selectors.append(y_k)
            new_vars.append(y_k)

        sum_sel = selectors[0]
        for k in range(1, n_disjuncts):
            sum_sel = sum_sel + selectors[k]

        new_cons.append(
            Constraint(
                body=sum_sel - _wrap(1.0),
                sense="==",
                rhs=0.0,
                name=f"_hull_select_{prefix}",
            )
        )

    # Refuse before doing any work if any body hides a variable from the
    # collector everything below is keyed on (checked per body -- see the
    # function's docstring for why the disjunction-wide union is not enough).
    _assert_hull_saw_every_variable(dc)

    # --- Nested disjunctions, leaves first (#1775 G2.1) ---
    # rows[k]: disjunct k's rows, its children's emitted rows included;
    # local[k]: the variables its children created, keyed by name.
    rows: list[list[Constraint]] = []
    local: list[dict[str, Variable]] = []
    for k, disjunct in enumerate(dc.disjuncts):
        rows_k: list[Constraint] = []
        local_k: dict[str, Variable] = {}
        for j, item in enumerate(disjunct):
            if isinstance(item, _DisjunctiveConstraint):
                own = (child_selectors or {}).get(id(item))
                child_vars, child_cons = _reformulate_disjunction_hull(
                    item,
                    model,
                    add_aux_binary,
                    eps=eps,
                    selectors=own,
                    prefix=f"{prefix}_d{k}_{item.name or j}",
                    child_selectors=child_selectors,
                )
                # A block's indicators -- the child's and, below it, its own
                # children's -- are zero whenever y_k is (their ``sum == parent``
                # rows), so they are disjunct k's own, like fresh selectors.
                local_k.update(
                    (v.name, v)
                    for v in [*_nested_block_selectors(item, child_selectors), *child_vars]
                )
                rows_k.extend(child_cons)
                new_vars.extend(child_vars)
            else:
                rows_k.append(item)
        rows.append(rows_k)
        local.append(local_k)

    # --- Collect all variables across all disjuncts ---
    # A child's variables are local to their disjunct, never disaggregated again --
    # unless another disjunct uses one too (a nested block's indicator in a
    # sibling's rows): no one copy is both, so it is shared, and disaggregated.
    used = [
        {n: v for con in rows_k for n, v in _collect_variables(con.body).items()} for rows_k in rows
    ]
    for k, local_k in enumerate(local):
        for n in [n for n in local_k if any(n in used[j] for j in range(n_disjuncts) if j != k)]:
            del local_k[n]
    local_names = {name for local_k in local for name in local_k}
    all_vars: dict[str, Variable] = {}
    for used_k in used:
        all_vars.update((n, v) for n, v in used_k.items() if n not in local_names)

    # --- Per-disjunct bounds and disaggregated variables ---
    # disjunct_bounds[k][var_name] = (dlb, dub). Only the disjunct's own rows bound
    # its copies: a child's rows hold under a child selector, not under y_k.
    disjunct_bounds: list[dict[str, tuple[float, float]]] = []
    for k, disjunct in enumerate(dc.disjuncts):
        db = _extract_disjunct_bounds(
            [c for c in disjunct if not isinstance(c, _DisjunctiveConstraint)], model
        )
        # Fill in global bounds for variables not constrained in this disjunct
        for vname, var in all_vars.items():
            if vname not in db:
                db[vname] = (float(np.min(var.lb)), float(np.max(var.ub)))
        disjunct_bounds.append(db)

    # #1617: every bound-linking row ``dlb*y_k <= v_{j,k} <= dub*y_k`` needs a FINITE
    # bound. At the unbounded sentinel (9.999e19) the row is ``v <= 1e20*y``: exact
    # only in exact arithmetic -- the integrality tolerance lets ``y = 1e-5`` buy
    # ``v <= 1e15`` -- and every LP/MILP engine refuses or mangles the coefficient
    # (HiGHS: ``|a_ij| >= 1e15`` -> kError, surfaced as a causeless ``error``).
    # Dropping the row instead is not exact either: an unselected disjunct's copy
    # then ranges over that disjunct's recession cone. Refuse, naming the variable,
    # exactly as big-M refuses a sentinel-sized M.
    for k, db in enumerate(disjunct_bounds):
        for vname, (dlb, dub) in db.items():
            if not (np.isfinite(dlb) and np.isfinite(dub)) or max(abs(dlb), abs(dub)) >= (
                _BIGM_SENTINEL
            ):
                side = "upper" if not np.isfinite(dub) or abs(dub) >= _BIGM_SENTINEL else "lower"
                raise ValueError(
                    f"GDP hull reformulation of disjunction {prefix!r}: variable {vname!r} "
                    f"has no finite {side} bound within disjunct {k} (bounds "
                    f"[{dlb:g}, {dub:g}]). The hull links each disaggregated copy to its "
                    "selector by that bound (v <= ub*y), so an unbounded variable "
                    "admits no exact hull. Give the variable a finite bound (e.g. "
                    "m.continuous(..., ub=...)) or bound it inside every disjunct."
                )

    # disagg[k][var_name] = disaggregated Variable v_{j,k}
    disagg: list[dict[str, Variable]] = []
    for k in range(n_disjuncts):
        disagg_k: dict[str, Variable] = {}
        for vname, var in all_vars.items():
            dlb, dub = disjunct_bounds[k][vname]
            # Disaggregated bounds: [min(dlb, 0), max(dub, 0)]
            v_lb = min(dlb, 0.0)
            v_ub = max(dub, 0.0)
            v_jk = Variable(
                model._fresh_name(f"_hull_{prefix}_v_{vname}_{k}"),  # #1799
                VarType.CONTINUOUS,
                var.shape,
                v_lb,
                v_ub,
                model,
            )
            disagg_k[vname] = v_jk
            new_vars.append(v_jk)
            model._register_generated_variable(v_jk)
        disagg.append(disagg_k)

    # --- Aggregation: x_j == sum_k(v_{j,k}) ---
    for vname, var in all_vars.items():
        agg_expr: Expression = disagg[0][vname]
        for k in range(1, n_disjuncts):
            agg_expr = agg_expr + disagg[k][vname]
        new_cons.append(
            Constraint(
                body=var - agg_expr,
                sense="==",
                rhs=0.0,
                name=f"_hull_agg_{prefix}_{vname}",
            )
        )

    # --- Bound linking: dlb * y_k <= v_{j,k} <= dub * y_k ---
    for k in range(n_disjuncts):
        y_k = selectors[k]
        for vname in all_vars:
            dlb, dub = disjunct_bounds[k][vname]
            v_jk = disagg[k][vname]
            # v_{j,k} <= dub * y_k  =>  v_{j,k} - dub * y_k <= 0
            new_cons.append(
                Constraint(
                    body=v_jk - _wrap(dub) * y_k,
                    sense="<=",
                    rhs=0.0,
                    name=f"_hull_ub_{prefix}_{vname}_{k}",
                )
            )
            # v_{j,k} >= dlb * y_k  =>  v_{j,k} - dlb * y_k >= 0
            new_cons.append(
                Constraint(
                    body=v_jk - _wrap(dlb) * y_k,
                    sense=">=",
                    rhs=0.0,
                    name=f"_hull_lb_{prefix}_{vname}_{k}",
                )
            )
    # A local w is zero whenever y_k is, so its bounds hold 0 (a copy's
    # [min(lb,0), max(ub,0)]; a selector's [0, 1]). Its bound links are implied
    # (see the docstring), so none is emitted here.
    for k in range(n_disjuncts):
        for wname, w in local[k].items():
            wlb, wub = float(np.min(w.lb)), float(np.max(w.ub))
            if not (wlb <= 0.0 <= wub):
                raise AssertionError(
                    f"hull of {prefix!r}: nested variable {wname!r} has bounds "
                    f"[{wlb:g}, {wub:g}], which exclude 0"
                )

    # --- Constraint reformulation per disjunct ---
    for k, rows_k in enumerate(rows):
        y_k = selectors[k]
        # The perspective divides every variable owned by disjunct k by yhat_k:
        # its copies of the shared variables, and its children's locals.
        zero_vars = {**all_vars, **local[k]}

        for j, con in enumerate(rows_k):
            cname = f"_hull_{prefix}_d{k}_c{j}"

            if _is_linear(con.body):
                # Linear f(x) = a^T x + b: hull gives a^T v_k + b * y_k
                # Substitute vars -> disagg vars and scale constants by y_k
                hull_body = _hull_linear_substitute(
                    con.body, {vname: disagg[k][vname] for vname in all_vars}, y_k
                )
                rhs_expr = _wrap(con.rhs) * y_k
            else:
                # Nonlinear: the Furman-Sawaya-Grossmann eps-perspective
                # (Furman2020 in docs/references.bib), for g(x) = body - rhs <= 0:
                #
                #     yhat * g(v_k / yhat) - eps * g(0) * (1 - y_k) <= 0,
                #     yhat = (1 - eps) * y_k + eps.
                #
                # Both pieces are load-bearing, and the previous form
                # (``yhat = y_k + eps`` with no g(0) term) had NEITHER, so it was
                # exact at NEITHER integer face:
                #
                #   y_k = 0: the disaggregated vars are pinned to 0 by the bound
                #     linking rows, so the body evaluates to eps * g(0), not 0.
                #     Whenever g(0) != 0 that is a HARD violation of a row the
                #     model states as an equality -- the "off" disjunct can never
                #     actually be off. The -eps*g(0)*(1-y_k) term cancels it
                #     exactly.
                #   y_k = 1: yhat was 1 + eps, so the row read
                #     g(v/(1+eps)) * (1+eps) -- an O(eps) DISTORTION of the
                #     selected disjunct's own constraint. Scaling eps into the
                #     y_k coefficient makes yhat exactly 1 there.
                #
                # #1043: this is not a tolerance nicety. The residuals make the
                # reformulated model infeasible in exact arithmetic, and the
                # rigorous machinery downstream (FBBT / the McCormick LP, which
                # must be exact to be sound) then proves it empty and reports
                # ``infeasible`` for a feasible model -- the worst-class error.
                # Measured on ``max x s.t. if_else(x>=0, exp(x)-1, log(-x+3)) <= 1,
                # x in [-10,10]``: at the true optimum x = log 2 the old rows are
                # violated by 1.0986e-08 = eps*log(3) (the off disjunct) and
                # 3.86e-09 (the on disjunct). Shrinking eps does not help --
                # the model is still exactly infeasible at eps = 1e-14 -- which is
                # why only cancelling the terms, not making them small, fixes it.
                g0 = _body_at_zero(con.body, zero_vars) - float(np.max(con.rhs))
                y_clamp = _wrap(1.0 - eps) * y_k + _wrap(eps)
                persp_map = {vname: disagg[k][vname] / y_clamp for vname in all_vars}
                persp_map.update({wname: w / y_clamp for wname, w in local[k].items()})
                subst_body = _substitute_vars(con.body, persp_map)
                hull_body = subst_body * y_clamp
                if g0 != 0.0:
                    hull_body = hull_body - _wrap(eps * g0) * (_wrap(1.0) - y_k)
                rhs_expr = _wrap(con.rhs) * y_clamp

            if con.sense == "<=":
                new_cons.append(
                    Constraint(
                        body=hull_body - rhs_expr,
                        sense="<=",
                        rhs=0.0,
                        name=cname,
                    )
                )
            elif con.sense == ">=":
                new_cons.append(
                    Constraint(
                        body=hull_body - rhs_expr,
                        sense=">=",
                        rhs=0.0,
                        name=cname,
                    )
                )
            elif con.sense == "==":
                new_cons.append(
                    Constraint(
                        body=hull_body - rhs_expr,
                        sense="<=",
                        rhs=0.0,
                        name=f"{cname}_le",
                    )
                )
                new_cons.append(
                    Constraint(
                        body=hull_body - rhs_expr,
                        sense=">=",
                        rhs=0.0,
                        name=f"{cname}_ge",
                    )
                )

    return new_vars, new_cons


class HullVariableCoverageError(ValueError):
    """A disjunct body holds a variable the hull walkers cannot see.

    Raised by :func:`_assert_hull_saw_every_variable`. See its docstring for why
    this is a refusal and not a best-effort reformulation.
    """


def _nested_block_selectors(
    dc: _DisjunctiveConstraint, child_selectors: Optional[dict[int, list[Variable]]]
) -> list[Variable]:
    """The indicators of ``dc`` and of every block nested below it (#1775 G2.1).

    Only a nested ``add_disjunction`` block has them (``child_selectors``); a
    disjunction hulled on fresh selectors returns those among its new variables.
    """
    out = list((child_selectors or {}).get(id(dc), []))
    for disjunct in dc.disjuncts:
        for item in disjunct:
            if isinstance(item, _DisjunctiveConstraint):
                out.extend(_nested_block_selectors(item, child_selectors))
    return out


def _assert_hull_saw_every_variable(dc: _DisjunctiveConstraint) -> None:
    """Cross-check ``_collect_variables`` against an *independent* DAG walker.

    Everything downstream of the collector in :func:`_reformulate_disjunction_hull`
    -- the disaggregated variables, the aggregation rows ``x = Σ_k v_k``, the bound
    linking rows ``dlb·y_k ≤ v_k ≤ dub·y_k``, the substitution map handed to
    :func:`_hull_linear_substitute` / :func:`_substitute_vars` -- is keyed on it.
    A variable the collector misses is therefore not merely under-disaggregated:
    it survives into the reformulated row **un-substituted**, so the disjunct
    body is imposed *globally* rather than under its selector.

    That is not hypothetical. PR #1150 widened :func:`_is_linear` to admit
    ``SumOverExpression`` while the collector still had no case for the node.
    ``all_vars`` came back empty, no disaggregated variable was created, and hull
    emitted ``Σ[3 terms] - (0 * y0) <= 0`` — the arm's own constraint, always on,
    with its selector coefficient collapsed to zero. On a model whose true
    minimum is −30.0 both ``auto`` and ``hull`` then returned
    ``status=optimal, objective=−3.0, bound=−3.0``: a dual bound **above** the
    optimum, i.e. a false certificate (CLAUDE.md §1).

    **The check is PER CONSTRAINT BODY, and that is load-bearing.** The first
    version of this guard compared the walkers' *disjunction-wide unions*, which
    made a per-body miss invisible whenever the missed variable was collected
    from any other constraint in the same disjunction — the common case, since
    the arms of a disjunction usually constrain the same variables. Found in
    review of PR #1159 with a 6-line repro that the union form passed silently::

        X = m.continuous("X", shape=(3,), lb=0.0, ub=10.0)
        A = np.array([[1.0, 1.0, 1.0]])
        m.either_or([[(A @ X)[0] - 3.0 <= 0.0], [X[0] >= 8.0]])
        m.minimize(-(X[0] + X[1] + X[2]))     # true minimum -30.0

    ``(A @ X)[0]`` is an ``IndexExpression`` over a ``MatMulExpression``.
    :func:`_is_linear` answers ``True`` for any ``IndexExpression`` without
    inspecting its base, so the body takes the *linear* route, while
    :func:`_collect_variables` returns ``{}`` for it. ``X`` reached ``all_vars``
    from the other arm, the union check passed, and hull certified ``-3.0``.
    Measured on ``main`` too: this false certificate predates #1154. Per body,
    the guard fires on it.

    The check is deliberately built on a walker this module does not own:
    :func:`discopt.modeling.core._iter_model_leaves`, which enumerates the
    model-owning leaves (``Variable`` / ``Parameter``) and already handles
    ``SumExpression``, ``SumOverExpression``, ``MatMulExpression`` and
    ``CustomCall``. A cross-check against the *same* walker would be a no-op
    (CLAUDE.md §6): if the collector cannot see a node, neither could the probe.

    Refuses loudly rather than falling back, per CLAUDE.md §3: there is no sound
    cheaper answer here — the alternative is exactly the silent wrong row above.
    It costs no model that reformulates today. On the nonlinear route
    :func:`_body_at_zero` already raises :class:`HullPerspectiveOriginError` for
    every node type that can trip this; on the linear route the only way in is an
    ``IndexExpression`` over one of them, which is the defect class itself.
    """
    from discopt.modeling.core import Parameter, _iter_model_leaves

    for k, disjunct in enumerate(dc.disjuncts):
        for j, con in enumerate(disjunct):
            if isinstance(con, _DisjunctiveConstraint):
                continue  # checked by its own hull call (#1775 G2.1)
            collected = _collect_variables(con.body)
            missed = sorted(
                {
                    leaf.name
                    for leaf in _iter_model_leaves(con.body)
                    if not isinstance(leaf, Parameter) and leaf.name not in collected
                }
            )
            if missed:
                cname = getattr(con, "name", None) or f"disjunct {k}, constraint {j}"
                raise HullVariableCoverageError(
                    f"GDP hull reformulation cannot see every variable in the body of "
                    f"{cname}: {missed} are reachable from that body's expression DAG "
                    "but were not collected from it, so they would be emitted "
                    "un-disaggregated and the disjunct's constraints would be imposed "
                    "globally (a false certificate -- see issue #1154). Reformulate "
                    "this disjunction with method='big-m', or express the body with "
                    "+/-/* so the hull walkers can take it apart."
                )


class HullPerspectiveOriginError(ValueError):
    """``g(0)`` is not finite, so the hull perspective cannot be formed.

    Raised by :func:`_body_at_zero`. See its docstring for why this is a refusal
    rather than a fallback value.
    """


def _body_at_zero(expr: Expression, all_vars: dict[str, Variable]) -> float:
    """Evaluate a disjunct constraint body with every disjunct variable at 0.

    The Furman-Sawaya-Grossmann perspective needs ``g(0)`` to cancel the residual
    the eps clamp leaves on a de-selected disjunct (#1043). ``g(0)`` is a property
    of the *function*, not of the box: at ``y_k = 0`` the bound-linking rows pin
    every disaggregated variable to exactly 0, so ``g(0)`` is the value the row
    actually takes there, whether or not 0 lies inside the disjunct's own bounds.

    Refuses loudly (CLAUDE.md §3) when the body is not finite at the origin --
    ``log(0)``, ``1/0``, ``sqrt`` of a negative -- because there is no sound
    numeric substitute. A fabricated finite ``g(0)`` would leave an uncancelled
    eps-scale residual on the off disjunct, which is exactly the false-infeasible
    class this function exists to close. The old formulation did not refuse: it
    emitted ``f(0) * eps`` = NaN/-inf straight into the row.
    """
    _UN: dict[str, Callable[[float], float]] = {
        "neg": lambda a: -a,
        "-": lambda a: -a,
        "abs": abs,
        "exp": np.exp,
        "log": np.log,
        "sqrt": np.sqrt,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "tanh": np.tanh,
    }
    _BIN: dict[str, Callable[[float, float], float]] = {
        "+": lambda a, b: a + b,
        "-": lambda a, b: a - b,
        "*": lambda a, b: a * b,
        "/": lambda a, b: a / b,
        "**": lambda a, b: a**b,
        "^": lambda a, b: a**b,
    }
    _NARY: dict[str, Callable[..., float]] = {"max": max, "min": min}

    def _walk(e: Expression) -> float:
        if isinstance(e, Variable):
            if e.name not in all_vars:
                raise HullPerspectiveOriginError(
                    f"variable {e.name!r} appears in a disjunct body but was not "
                    "collected as a disjunct variable"
                )
            return 0.0
        if isinstance(e, Constant):
            return float(np.max(e.value))
        if isinstance(e, IndexExpression):
            return _walk(e.base)
        if isinstance(e, UnaryOp):
            if e.op not in _UN:
                raise HullPerspectiveOriginError(f"unsupported unary op {e.op!r}")
            return float(_UN[e.op](_walk(e.operand)))
        if isinstance(e, BinaryOp):
            if e.op not in _BIN:
                raise HullPerspectiveOriginError(f"unsupported binary op {e.op!r}")
            return float(_BIN[e.op](_walk(e.left), _walk(e.right)))
        sum_terms = _sumover_terms(e)
        if sum_terms is not None:
            # g(0) of a sum is the sum of the terms' g(0); left-folded to match
            # the equivalent ``t1 + ... + tn`` chain bit for bit.
            total = 0.0
            for term in sum_terms:
                total = total + _walk(term)
            return float(total)
        if isinstance(e, FunctionCall):
            args = [_walk(a) for a in e.args]
            if e.func_name in _NARY:
                return float(_NARY[e.func_name](*args))
            if e.func_name not in _UN or len(args) != 1:
                raise HullPerspectiveOriginError(f"unsupported function {e.func_name!r}")
            return float(_UN[e.func_name](args[0]))
        raise HullPerspectiveOriginError(f"unsupported expression node {type(e).__name__}")

    with np.errstate(all="raise"):
        try:
            val = _walk(expr)
        except (FloatingPointError, ValueError, ZeroDivisionError, OverflowError) as exc:
            if isinstance(exc, HullPerspectiveOriginError):
                raise
            raise HullPerspectiveOriginError(
                f"the disjunct body is not finite at the origin ({exc}); the hull "
                "perspective needs g(0) to cancel the eps residual on a de-selected "
                "disjunct. Reformulate this disjunction with method='big-m'."
            ) from exc
    if not np.isfinite(val):
        raise HullPerspectiveOriginError(
            f"the disjunct body evaluates to {val} at the origin; the hull "
            "perspective needs a finite g(0) to cancel the eps residual on a "
            "de-selected disjunct. Reformulate this disjunction with method='big-m'."
        )
    return val


def _hull_linear_substitute(
    expr: Expression,
    var_map: dict[str, Expression],
    y_k: Expression,
) -> Expression:
    """Substitute variables and scale constants by y_k for hull linear reform.

    For a linear expression ``a*x + b``, returns ``a*v_k + b*y_k`` where
    ``v_k`` is the disaggregated variable from *var_map*.
    """
    if isinstance(expr, Variable):
        return var_map.get(expr.name, expr)
    if isinstance(expr, Constant):
        result: Expression = expr * y_k
        return result
    if isinstance(expr, IndexExpression):
        if isinstance(expr.base, Variable) and expr.base.name in var_map:
            new_base = var_map[expr.base.name]
            return IndexExpression(new_base, expr.index)
        return expr
    if isinstance(expr, BinaryOp):
        if expr.op in ("+", "-"):
            new_left = _hull_linear_substitute(expr.left, var_map, y_k)
            new_right = _hull_linear_substitute(expr.right, var_map, y_k)
            return BinaryOp(expr.op, new_left, new_right)
        if expr.op == "*":
            # For linear expressions, one side must be a constant
            if isinstance(expr.left, Constant):
                # const * linear_expr → const * hull_substitute(linear_expr)
                new_right = _hull_linear_substitute(expr.right, var_map, y_k)
                return BinaryOp("*", expr.left, new_right)
            if isinstance(expr.right, Constant):
                new_left = _hull_linear_substitute(expr.left, var_map, y_k)
                return BinaryOp("*", new_left, expr.right)
        if expr.op == "/":
            if isinstance(expr.right, Constant):
                new_left = _hull_linear_substitute(expr.left, var_map, y_k)
                return BinaryOp("/", new_left, expr.right)
    if isinstance(expr, UnaryOp):
        if expr.op in ("neg", "-"):
            new_operand = _hull_linear_substitute(expr.operand, var_map, y_k)
            return UnaryOp(expr.op, new_operand)
    terms = _sumover_terms(expr)
    if terms is not None:
        # Term by term: each term contributes ``a_i v_k + b_i y_k``, so the sum
        # contributes ``a v_k + b y_k`` -- the same expression the equivalent
        # ``t1 + ... + tn`` chain produces through the "+" branch above.
        return SumOverExpression([_hull_linear_substitute(t, var_map, y_k) for t in terms])
    # Fallback
    return _substitute_vars(expr, var_map)


# ── Hull reformulation helpers ──


def _collect_variables(expr: Expression) -> dict[str, Variable]:
    """Walk expression DAG and return Variable nodes keyed by name.

    Variables are deduplicated by name (Variable is not hashable due to
    ``__eq__`` returning a Constraint).
    """
    found: dict[str, Variable] = {}
    # #1565: a shared subexpression is walked once. Re-walking it can only
    # re-assign names already in ``found`` (same name, same Variable), which
    # changes neither the contents nor the insertion order, so skipping it is
    # exact. Holding the node keeps its ``id`` from being recycled mid-walk.
    seen: dict[int, Expression] = {}

    def _walk(e: Expression) -> None:
        if id(e) in seen:
            return
        seen[id(e)] = e
        if isinstance(e, Variable):
            found[e.name] = e
        elif isinstance(e, IndexExpression):
            if isinstance(e.base, Variable):
                found[e.base.name] = e.base
            else:
                _walk(e.base)
        elif isinstance(e, BinaryOp):
            _walk(e.left)
            _walk(e.right)
        elif isinstance(e, UnaryOp):
            _walk(e.operand)
        elif isinstance(e, FunctionCall):
            for arg in e.args:
                _walk(arg)
        else:
            terms = _sumover_terms(e)
            if terms is not None:
                for term in terms:
                    _walk(term)

    _walk(expr)
    return found


def _is_linear(expr: Expression) -> bool:
    """Check if expression is linear in its variables (conservative).

    Returns True only for expressions involving +, -, and * where at least
    one operand of every multiplication is a constant. Returns False for
    any nonlinear operation (**, FunctionCall, bilinear terms).
    False negatives are safe (they cause fallback to perspective functions).
    """
    # #1814: an explicit worklist, not one Python frame per level. A loop-built
    # ``x[0] + ... + x[2499]`` overflowed here when ``Model.complementarity`` asked
    # whether its operand needs a lift, at build time, outside any solve-time
    # recursion gate. The predicate is "every node reached passes its local test",
    # so the visiting order cannot change the answer.
    stack = [expr]
    seen: set[int] = set()
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, (Variable, Constant, IndexExpression)):
            continue
        if isinstance(node, UnaryOp):
            if node.op not in ("neg", "-"):
                return False
            stack.append(node.operand)
            continue
        if isinstance(node, BinaryOp):
            if node.op in ("+", "-"):
                stack.extend((node.left, node.right))
            elif node.op == "*":
                # Linear only if at least one side is a constant
                if not (isinstance(node.left, Constant) or isinstance(node.right, Constant)):
                    return False
                stack.extend((node.left, node.right))
            elif node.op == "/":
                # x / constant is linear
                if not isinstance(node.right, Constant):
                    return False
                stack.append(node.left)
            else:
                # ** is always nonlinear
                return False
            continue
        if isinstance(node, FunctionCall):
            return False
        # #1154: the indexed summation Σ[t0, t1, ...] IS linear when every term is,
        # and is admitted as such (``DISCOPT_GDP_SUMOVER``, default ON) — but only
        # because the whole walker family moved with it. The ORDER is the lesson.
        #
        # ``_is_linear`` is a shared gate with ~10 consumers, and several read a True
        # as "and I can therefore take this apart". #1039's first attempt widened
        # this predicate ALONE, while the hull substitution family
        # (``_hull_linear_substitute`` → ``_substitute_vars`` → ``_collect_variables``)
        # and ``_bound_expression`` still had no case for the node. ``all_vars`` came
        # back empty, so hull emitted ``Σ[3 terms] - (0 * y0) <= 0`` — the disjunct
        # body imposed GLOBALLY, its selector coefficient collapsed to zero and the
        # disaggregated variables never created. That cut off the other mode and
        # certified ``optimal`` at -3.0 against a true -30.0: a dual bound ABOVE the
        # minimum, where the conservative False had produced a loud refusal. #1150
        # reverted it; #1154 shipped it with the family complete, plus
        # ``_assert_hull_saw_every_variable``, which now refuses loudly rather than
        # emitting that row for ANY node the collector cannot see.
        #
        # Two things stay true regardless. A caller that only needs "can this body
        # become one LP row" should ask ``_extract_body_coeffs`` directly — it returns
        # the row as the witness, so it cannot promise more than it delivers. And
        # widening this predicate is bound-changing (it moves the FBBT structural mask
        # at ``solver.py:3068`` and the aux-lift gate at ``factorable_reform.py:684``),
        # which is why it ships behind a flag; the §5 panel that graduated it
        # default-ON is ``docs/dev/issue-1154-gdp-sumover-panel-2026-09-04.md``.
        terms = _sumover_terms(node)
        if terms is None:
            return False
        # A sum is linear iff every term is; an empty sum is the constant 0.
        stack.extend(terms)
    return True


def _substitute_vars(
    expr: Expression,
    var_map: dict[str, Expression],
) -> Expression:
    """Return a copy of *expr* with variables renamed per *var_map*.

    *var_map* maps variable **names** to replacement expressions.
    Nodes not in the map are returned unchanged.
    """
    if isinstance(expr, Variable):
        return var_map.get(expr.name, expr)
    if isinstance(expr, IndexExpression):
        if isinstance(expr.base, Variable) and expr.base.name in var_map:
            new_base = var_map[expr.base.name]
            return IndexExpression(new_base, expr.index)
        new_base = _substitute_vars(expr.base, var_map)
        if new_base is expr.base:
            return expr
        return IndexExpression(new_base, expr.index)
    if isinstance(expr, BinaryOp):
        new_left = _substitute_vars(expr.left, var_map)
        new_right = _substitute_vars(expr.right, var_map)
        if new_left is expr.left and new_right is expr.right:
            return expr
        return BinaryOp(expr.op, new_left, new_right)
    if isinstance(expr, UnaryOp):
        new_operand = _substitute_vars(expr.operand, var_map)
        if new_operand is expr.operand:
            return expr
        return UnaryOp(expr.op, new_operand)
    if isinstance(expr, FunctionCall):
        new_args = tuple(_substitute_vars(a, var_map) for a in expr.args)
        if all(n is o for n, o in zip(new_args, expr.args)):
            return expr
        return FunctionCall(expr.func_name, *new_args)
    terms = _sumover_terms(expr)
    if terms is not None:
        new_terms = [_substitute_vars(t, var_map) for t in terms]
        if all(n is o for n, o in zip(new_terms, terms)):
            return expr
        return SumOverExpression(new_terms)
    # Constant or unknown — pass through
    return expr


def _whole_variable(expr) -> "Variable | None":
    """The `Variable` this expression denotes *in its entirety*, else ``None``.

    The bounds this module extracts are keyed by variable **name**, so they apply
    to every component of an array variable. A constraint on a *slice* of that
    array therefore must not produce one: in the hull reformulation the extracted
    bound caps every disaggregated component, so a bound taken from ``x[0]`` and
    applied to all of ``x`` cuts feasible points.

    That was a live false-optimum bug, not a hypothetical. With ``x`` of shape
    ``(3,)`` in ``[0, 10]``::

        m.either_or([[x[0] <= 1.0], [x[0] >= 9.0]])
        m.subject_to(x[0] <= 1.0)      # pin the first disjunct
        m.maximize(x[1] + x[2])        # true optimum 20.0

    the hull reformulation capped ``x[1]`` and ``x[2]`` at 1 as well and returned
    ``status=optimal, objective=2.0`` -- a wrong certificate. (``big-m`` and
    ``mbigm`` were unaffected; they do not use these bounds this way.)

    So: accept a bare `Variable`, and accept an `IndexExpression` only when it
    selects the whole variable (its shape equals the base's). Anything narrower
    yields ``None`` and the caller falls back to the variable's global bounds --
    a weaker relaxation, and a valid one.
    """
    if isinstance(expr, Variable):
        return expr
    if isinstance(expr, IndexExpression) and isinstance(expr.base, Variable):
        if tuple(getattr(expr, "shape", ())) == tuple(getattr(expr.base, "shape", ())):
            return expr.base
    return None


def _extract_disjunct_bounds(
    disjunct: list[Constraint],
    model: Model,
) -> dict[str, tuple[float, float]]:
    """Extract per-variable bounds implied by simple constraints in a disjunct.

    Scans for single-variable bound patterns (``x <= c``, ``x >= c``,
    ``x == c``) and returns the tightest implied bounds, starting from the
    variable's global bounds.

    Parameters
    ----------
    disjunct : list[Constraint]
        Constraints in one arm of a disjunction.
    model : Model
        Model with global variable bound information.

    Returns
    -------
    dict[str, tuple[float, float]]
        Mapping from variable name to (lb, ub).
    """
    from discopt._relax.scalarize import scalar_leaf_value

    bounds: dict[str, list[float]] = {}

    def _init_var(v: Variable) -> None:
        if v.name not in bounds:
            bounds[v.name] = [float(np.min(v.lb)), float(np.max(v.ub))]

    def _scalar_const(e: Expression) -> float | None:
        # #1825: only a scalar constant is one bound. An array-valued one makes the
        # row several bounds (``float()`` on it raised a TypeError); such a row is
        # skipped, which leaves the variable's global bounds -- looser, so sound.
        return scalar_leaf_value(e) if isinstance(e, Constant) else None

    for con in disjunct:
        # Identify single-variable body, possibly with constant offset
        body = con.body
        var: Variable | None = None
        offset = 0.0  # effective constraint: var (sense) rhs - offset

        if (whole := _whole_variable(body)) is not None:
            var = whole
        elif isinstance(body, BinaryOp) and body.op == "-":
            # Pattern: var - const <= 0  =>  var <= const
            if (lvar := _whole_variable(body.left)) is not None and (
                (rc := _scalar_const(body.right)) is not None
            ):
                var = lvar
                offset = -rc
            # Pattern: const - var <= 0  =>  var >= const
            elif (lc := _scalar_const(body.left)) is not None and (
                (rvar := _whole_variable(body.right)) is not None
            ):
                var = rvar
                # const - var (sense) 0  =>  -var (sense) -const  =>  var (flip) const
                # We handle this by negating and flipping sense below
                offset = lc
                # For "const - var <= 0" => var >= const
                # For "const - var >= 0" => var <= const
                _init_var(var)
                effective_rhs = offset  # the constant value
                if con.sense == "<=":
                    # const - var <= 0 => var >= const
                    bounds[var.name][0] = max(bounds[var.name][0], effective_rhs)
                elif con.sense == ">=":
                    # const - var >= 0 => var <= const
                    bounds[var.name][1] = min(bounds[var.name][1], effective_rhs)
                elif con.sense == "==":
                    bounds[var.name][0] = max(bounds[var.name][0], effective_rhs)
                    bounds[var.name][1] = min(bounds[var.name][1], effective_rhs)
                continue
        elif isinstance(body, BinaryOp) and body.op == "+":
            # Pattern: var + const <= 0  =>  var <= -const
            if (lvar := _whole_variable(body.left)) is not None and (
                (rc := _scalar_const(body.right)) is not None
            ):
                var = lvar
                offset = rc

        if var is None:
            continue

        # effective: var (sense) rhs - offset
        rhs = con.rhs - offset
        _init_var(var)

        if con.sense == "<=":
            bounds[var.name][1] = min(bounds[var.name][1], rhs)
        elif con.sense == ">=":
            bounds[var.name][0] = max(bounds[var.name][0], rhs)
        elif con.sense == "==":
            bounds[var.name][0] = max(bounds[var.name][0], rhs)
            bounds[var.name][1] = min(bounds[var.name][1], rhs)

    return {name: (lb_ub[0], lb_ub[1]) for name, lb_ub in bounds.items()}


# ── SOS constraint reformulation ──


def _reformulate_sos(
    sc: _SOSConstraint,
    model: Model,
    add_aux_binary,
) -> list[Constraint]:
    """Reformulate SOS Type 1 or Type 2 constraint via binary indicators.

    SOS1: At most one variable x_i can be nonzero.
      - Introduce binary z_i for each variable
      - x_i <= ub_i * z_i (linking upper bound)
      - x_i >= lb_i * z_i (linking lower bound, handles negative vars)
      - sum(z_i) <= 1

    SOS2: At most two *adjacent* variables can be nonzero.
      - Introduce binary z_i for each variable
      - x_i <= ub_i * z_i
      - x_i >= lb_i * z_i
      - sum(z_i) <= 2
      - z_i + z_j <= 1 for all non-adjacent pairs |i-j| > 1
    """
    new_cons = []
    n = len(sc.variables)

    # Create indicator binaries
    indicators = []
    for i in range(n):
        z_i = add_aux_binary(f"sos{sc.sos_type}_{sc.name or 'anon'}_{i}")
        indicators.append(z_i)

    # Linking constraints: x_i <= ub_i * z_i and x_i >= lb_i * z_i
    for i in range(n):
        v = sc.variables[i]
        # Extract bounds — v may be a Variable or IndexExpression
        lo_v, hi_v = _bound_expression(v, model)
        ub_val = hi_v
        lb_val = lo_v

        # GDP-1 (#413): the linking constraint ``x_i <= ub_i * z_i`` uses
        # ``ub_i`` as the value x_i may take when z_i=1. A too-small ``ub_i``
        # (e.g. clamping an infinite bound to ``_DEFAULT_BIG_M``) would cap the
        # variable below its true range and cut feasible SOS points — a false
        # certificate. A large *finite* bound is a valid cap; a truly infinite
        # bound has no valid finite cap, so refuse loudly instead of clamping.
        vname = getattr(v, "name", None) or f"variable {i}"
        if not np.isfinite(ub_val):
            raise ValueError(
                f"SOS{sc.sos_type} reformulation cannot bound {vname!r} from "
                f"above (upper bound is infinite), so the linking big-M would "
                f"be invalid and could cut feasible points. Add a finite upper "
                f"bound to this variable."
            )
        if not np.isfinite(lb_val):
            raise ValueError(
                f"SOS{sc.sos_type} reformulation cannot bound {vname!r} from "
                f"below (lower bound is infinite), so the linking big-M would "
                f"be invalid and could cut feasible points. Add a finite lower "
                f"bound to this variable."
            )

        # x_i - ub_i * z_i <= 0
        new_cons.append(
            Constraint(
                body=v - _wrap(ub_val) * indicators[i],
                sense="<=",
                rhs=0.0,
                name=f"_sos{sc.sos_type}_{sc.name}_ub_{i}" if sc.name else None,
            )
        )

        # x_i - lb_i * z_i >= 0  =>  x_i >= lb_i * z_i
        if lb_val < 0:
            new_cons.append(
                Constraint(
                    body=v - _wrap(lb_val) * indicators[i],
                    sense=">=",
                    rhs=0.0,
                    name=f"_sos{sc.sos_type}_{sc.name}_lb_{i}" if sc.name else None,
                )
            )

    if sc.sos_type == 1:
        # sum(z_i) <= 1
        if n == 1:
            sum_z = indicators[0]
        else:
            sum_z = indicators[0]
            for i in range(1, n):
                sum_z = sum_z + indicators[i]
        new_cons.append(
            Constraint(
                body=sum_z - _wrap(1.0),
                sense="<=",
                rhs=0.0,
                name=f"_sos1_{sc.name}_sum" if sc.name else None,
            )
        )

    elif sc.sos_type == 2:
        # sum(z_i) <= 2
        if n >= 2:
            sum_z = indicators[0]
            for i in range(1, n):
                sum_z = sum_z + indicators[i]
            new_cons.append(
                Constraint(
                    body=sum_z - _wrap(2.0),
                    sense="<=",
                    rhs=0.0,
                    name=f"_sos2_{sc.name}_sum" if sc.name else None,
                )
            )

            # Non-adjacency: z_i + z_j <= 1 for |i - j| > 1
            for i in range(n):
                for j in range(i + 2, n):
                    new_cons.append(
                        Constraint(
                            body=indicators[i] + indicators[j] - _wrap(1.0),
                            sense="<=",
                            rhs=0.0,
                            name=(f"_sos2_{sc.name}_nonadj_{i}_{j}" if sc.name else None),
                        )
                    )

    return new_cons


# ---------------------------------------------------------------------------
# Logical constraint linearization
# ---------------------------------------------------------------------------


def _get_binary_var(expr: LogicalExpression):
    """Extract the underlying Variable from a BooleanVar or indexed BooleanVar."""
    if isinstance(expr, BooleanVar):
        return expr.variable
    raise TypeError(f"Expected BooleanVar as leaf, got {type(expr).__name__}")


def _to_nnf(expr: LogicalExpression) -> LogicalExpression:
    """Convert a logical expression to Negation Normal Form (NNF).

    Pushes negation inward using De Morgan's laws. Eliminates Implies and
    Equivalent by rewriting them.
    """
    if isinstance(expr, BooleanVar):
        return expr
    elif isinstance(expr, LogicalNot):
        inner = expr.operand
        if isinstance(inner, BooleanVar):
            return expr  # literal: ~Y is already NNF
        elif isinstance(inner, LogicalNot):
            return _to_nnf(inner.operand)  # double negation
        elif isinstance(inner, LogicalAnd):
            # De Morgan: ~(A & B) = ~A | ~B
            return _to_nnf(LogicalOr(LogicalNot(inner.left), LogicalNot(inner.right)))
        elif isinstance(inner, LogicalOr):
            # De Morgan: ~(A | B) = ~A & ~B
            return _to_nnf(LogicalAnd(LogicalNot(inner.left), LogicalNot(inner.right)))
        elif isinstance(inner, LogicalImplies):
            # ~(A -> B) = A & ~B
            return _to_nnf(LogicalAnd(inner.antecedent, LogicalNot(inner.consequent)))
        elif isinstance(inner, LogicalEquivalent):
            # ~(A <-> B) = (A & ~B) | (~A & B)
            return _to_nnf(
                LogicalOr(
                    LogicalAnd(inner.left, LogicalNot(inner.right)),
                    LogicalAnd(LogicalNot(inner.left), inner.right),
                )
            )
        elif isinstance(inner, LogicalAtLeast):
            # ~(sum >= k)  ≡  sum <= k-1  ≡  atmost(k-1, ops)
            return LogicalAtMost(inner.k - 1, inner.operands)
        elif isinstance(inner, LogicalAtMost):
            # ~(sum <= k)  ≡  sum >= k+1  ≡  atleast(k+1, ops)
            return LogicalAtLeast(inner.k + 1, inner.operands)
        elif isinstance(inner, LogicalExactly):
            # ~(sum == k)  ≡  sum <= k-1  |  sum >= k+1
            return _to_nnf(
                LogicalOr(
                    LogicalAtMost(inner.k - 1, inner.operands),
                    LogicalAtLeast(inner.k + 1, inner.operands),
                )
            )
        else:
            return expr
    elif isinstance(expr, LogicalAnd):
        return LogicalAnd(_to_nnf(expr.left), _to_nnf(expr.right))
    elif isinstance(expr, LogicalOr):
        return LogicalOr(_to_nnf(expr.left), _to_nnf(expr.right))
    elif isinstance(expr, LogicalImplies):
        # A -> B  ≡  ~A | B
        return _to_nnf(LogicalOr(LogicalNot(expr.antecedent), expr.consequent))
    elif isinstance(expr, LogicalEquivalent):
        # A <-> B  ≡  (A -> B) & (B -> A)
        return _to_nnf(
            LogicalAnd(
                LogicalImplies(expr.left, expr.right),
                LogicalImplies(expr.right, expr.left),
            )
        )
    else:
        return expr


def _collect_literals(clause: LogicalExpression) -> list[tuple] | None:
    """Collect literals from a disjunction (Or-tree) of literals.

    Returns a list of (Variable, positive: bool) tuples, or None if the
    expression is not a flat clause.
    """
    if isinstance(clause, BooleanVar):
        return [(_get_binary_var(clause), True)]
    elif isinstance(clause, LogicalNot) and isinstance(clause.operand, BooleanVar):
        return [(_get_binary_var(clause.operand), False)]
    elif isinstance(clause, LogicalOr):
        left = _collect_literals(clause.left)
        right = _collect_literals(clause.right)
        if left is not None and right is not None:
            return left + right
    return None


def _nnf_to_cnf_clauses(expr: LogicalExpression, model: Model, add_aux_binary) -> list[list[tuple]]:
    """Convert NNF expression to CNF clauses.

    Each clause is a list of (Variable, positive: bool) literals.
    Uses Tseitin transformation for And nodes inside Or nodes.
    """
    # Literal
    if isinstance(expr, BooleanVar):
        return [[(_get_binary_var(expr), True)]]
    elif isinstance(expr, LogicalNot) and isinstance(expr.operand, BooleanVar):
        return [[(_get_binary_var(expr.operand), False)]]

    # Conjunction: collect clauses from both sides
    elif isinstance(expr, LogicalAnd):
        left_clauses = _nnf_to_cnf_clauses(expr.left, model, add_aux_binary)
        right_clauses = _nnf_to_cnf_clauses(expr.right, model, add_aux_binary)
        return left_clauses + right_clauses

    # Disjunction: try to collect flat literals first
    elif isinstance(expr, LogicalOr):
        literals = _collect_literals(expr)
        if literals is not None:
            return [literals]
        # Not flat — use Tseitin: introduce auxiliary for each non-literal child
        left_lit = _tseitin_var(expr.left, model, add_aux_binary)
        right_lit = _tseitin_var(expr.right, model, add_aux_binary)
        return [left_lit + right_lit]

    # Cardinality atom nested inside a boolean combination: introduce a
    # Tseitin auxiliary constrained to imply the atom, then emit the
    # single-literal clause forcing that auxiliary true. Silently returning
    # [] here would widen the feasible set (issue #750).
    elif isinstance(expr, (LogicalAtLeast, LogicalAtMost, LogicalExactly)):
        return [_tseitin_var(expr, model, add_aux_binary)]

    # No expression should reach here in NNF. Refuse loudly rather than
    # silently dropping the constraint and widening the feasible set.
    raise NotImplementedError(
        f"Cannot linearize logical sub-expression of type "
        f"{type(expr).__name__!r} during GDP reformulation; refusing to "
        f"silently drop it (would widen the feasible set). See issue #750."
    )


def _tseitin_var(expr: LogicalExpression, model: Model, add_aux_binary) -> list[tuple]:
    """Create a Tseitin auxiliary for a sub-expression, return its literal.

    Returns a list of literals (usually just one) that represent this
    sub-expression, and adds the defining clauses as constraints on the model.
    """
    # If already a literal, just return it
    if isinstance(expr, BooleanVar):
        return [(_get_binary_var(expr), True)]
    if isinstance(expr, LogicalNot) and isinstance(expr.operand, BooleanVar):
        return [(_get_binary_var(expr.operand), False)]

    # Create auxiliary binary: aux -> expr
    aux = add_aux_binary("_tseitin")

    # Cardinality atom: emit the big-M implication aux -> (cardinality)
    # directly; it has no CNF-clause representation.
    if isinstance(expr, (LogicalAtLeast, LogicalAtMost, LogicalExactly)):
        for cons in _cardinality_implication_constraints(aux, expr):
            model._constraints.append(cons)
        return [(aux, True)]

    # Get the sub-clauses for expr
    sub_clauses = _nnf_to_cnf_clauses(expr, model, add_aux_binary)

    # For each clause C_i in the CNF of expr:
    #   aux -> C_i  (i.e., ~aux | C_i)  means if aux=1, all clauses hold
    for clause in sub_clauses:
        linear_cons = _clause_to_constraint([(aux, False)] + clause)
        model._constraints.append(linear_cons)

    # For the reverse (C_1 & C_2 & ... -> aux), we add:
    #   For each clause C_i: if all C_i satisfied, aux=1
    #   This is complex for general CNF; for soundness we just add aux=1
    #   implication. The full Tseitin encoding is only needed for equivalence.
    #   Since we only need aux -> expr (one direction) for the clause we're
    #   building, the above is sufficient for correctness.

    return [(aux, True)]


def _cardinality_implication_constraints(
    aux, expr: "LogicalAtLeast | LogicalAtMost | LogicalExactly"
) -> list[Constraint]:
    """Build linear constraints encoding ``aux -> <cardinality atom>``.

    Used when a ``LogicalAtLeast`` / ``LogicalAtMost`` / ``LogicalExactly``
    atom appears nested inside a boolean combination and is lowered via a
    Tseitin auxiliary ``aux``. Only the forward implication is needed: the
    auxiliary occurs positively in exactly one clause, so it is existentially
    chosen, and ``aux -> atom`` is sufficient for soundness (see issue #750).

    Big-M form (``n`` = number of operands, ``s`` = sum of operand binaries):
      atleast(k):  s >= k*aux           (aux=0 => trivially true)
      atmost(k):   s <= n - (n-k)*aux   (aux=0 => s <= n, trivially true)
      exactly(k):  both of the above.
    """
    operands = expr.operands
    n = len(operands)
    var_sum = _wrap(0.0)
    for op in operands:
        var_sum = var_sum + _get_binary_var(op)

    constraints: list[Constraint] = []
    if isinstance(expr, (LogicalAtLeast, LogicalExactly)):
        # s - k*aux >= 0
        constraints.append(
            Constraint(
                body=var_sum - _wrap(float(expr.k)) * aux,
                sense=">=",
                rhs=0.0,
            )
        )
    if isinstance(expr, (LogicalAtMost, LogicalExactly)):
        # s + (n-k)*aux - n <= 0
        constraints.append(
            Constraint(
                body=var_sum + _wrap(float(n - expr.k)) * aux - _wrap(float(n)),
                sense="<=",
                rhs=0.0,
            )
        )
    return constraints


def _clause_to_constraint(clause: list[tuple], name: str | None = None) -> Constraint:
    """Convert a CNF clause to a linear constraint.

    Clause: [(var1, True), (var2, False), ...] meaning (var1 | ~var2 | ...)
    Linearization: sum(positive vars) + sum(1 - negative vars) >= 1
    """
    pos_sum = _wrap(0.0)
    n_neg = 0
    for var, positive in clause:
        if positive:
            pos_sum = pos_sum + var
        else:
            pos_sum = pos_sum + (_wrap(1.0) - var)
            n_neg += 1

    return Constraint(
        body=pos_sum - _wrap(1.0),
        sense=">=",
        rhs=0.0,
        name=name,
    )


def _logical_to_linear(
    lc: "_LogicalConstraint",
    model: Model,
    add_aux_binary,
) -> list[Constraint]:
    """Convert a logical constraint to linear constraints via CNF.

    Handles special cases (AtLeast, AtMost, Exactly) directly as
    linear sums. General propositional formulas go through NNF → CNF.
    """
    expr = lc.expression
    prefix = lc.name or "_logic"

    # Special cases: cardinality constraints
    if isinstance(expr, LogicalAtLeast):
        var_sum = _wrap(0.0)
        for op in expr.operands:
            var_sum = var_sum + _get_binary_var(op)
        return [
            Constraint(
                body=var_sum - _wrap(float(expr.k)),
                sense=">=",
                rhs=0.0,
                name=f"{prefix}_atleast",
            )
        ]

    if isinstance(expr, LogicalAtMost):
        var_sum = _wrap(0.0)
        for op in expr.operands:
            var_sum = var_sum + _get_binary_var(op)
        return [
            Constraint(
                body=var_sum - _wrap(float(expr.k)),
                sense="<=",
                rhs=0.0,
                name=f"{prefix}_atmost",
            )
        ]

    if isinstance(expr, LogicalExactly):
        var_sum = _wrap(0.0)
        for op in expr.operands:
            var_sum = var_sum + _get_binary_var(op)
        return [
            Constraint(
                body=var_sum - _wrap(float(expr.k)),
                sense="==",
                rhs=0.0,
                name=f"{prefix}_exactly",
            )
        ]

    # General case: NNF → CNF → linear
    nnf = _to_nnf(expr)
    clauses = _nnf_to_cnf_clauses(nnf, model, add_aux_binary)

    result = []
    for i, clause in enumerate(clauses):
        name_i = f"{prefix}_{i}" if lc.name else None
        result.append(_clause_to_constraint(clause, name=name_i))
    return result
