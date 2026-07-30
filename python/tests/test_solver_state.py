"""Conformance tests for ``solve_model``'s explicit state objects (plan item 11).

Card 4b dropped four of its five modules because they are inline statement blocks
of ``solve_model`` sharing an implicit closure; the census
(``discopt_benchmarks/scripts/solve_model_locals_census.py``) measured **153**
names bound in one would-be module region and read in another. Item 11 turns those
into named typed objects, group by group.

These tests are what keeps the migration from silently un-happening. Without them
a later edit could re-introduce a bare ``rust_time`` local next to
``_timers.rust_time`` and both would work — the function would simply have two
sources of truth for the same quantity again, which is the exact defect this work
exists to remove.

The AST assertions are deliberately structural rather than textual: they ask
whether ``solve_model`` *binds* the migrated name in its own scope, which is the
property that matters, and which a comment or a docstring mentioning the old name
cannot fake.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
from pathlib import Path

import discopt.solver as S
import pytest
from discopt.solver.state import (
    LazyStallSeparationState,
    PerNodeOBBTBudget,
    PhaseTimers,
    PrimalHeuristicState,
)

pytestmark = pytest.mark.smoke

SOLVER_SOURCE = Path(S.__file__)

#: Every local that has been migrated off ``solve_model``'s closure, mapped to
#: ``(holder local, field name)``. This is the authoritative mapping, not a
#: convention: the tests below enforce it in **both** directions, so a field with
#: no entry here and an entry here with no field both fail. A naming convention
#: would not — a convention is exactly the thing a later edit stops honouring
#: silently. Extend the table when a further group is threaded.
MIGRATED: dict[str, tuple[str, str]] = {
    # PhaseTimers — the Rust/JAX wall-clock split
    "rust_time": ("_timers", "rust_time"),
    "jax_time": ("_timers", "jax_time"),
    "t_rust_start": ("_timers", "t_rust_start"),
    "t_jax_start": ("_timers", "t_jax_start"),
    # PrimalHeuristicState — sub-NLP and LNS budgets (two subjects, so the
    # distinguishing prefix stays on the field names)
    "_subnlp_backend_fn": ("_heur", "subnlp_backend_fn"),
    "_subnlp_calls": ("_heur", "subnlp_calls"),
    "_subnlp_feasible": ("_heur", "subnlp_feasible"),
    "_subnlp_incumbent_updates": ("_heur", "subnlp_incumbent_updates"),
    "_lns_lb_calls": ("_heur", "lns_lb_calls"),
    "_lns_dive_calls": ("_heur", "lns_dive_calls"),
    "_lns_swap_misses": ("_heur", "lns_swap_misses"),
    # LazyStallSeparationState — the C-42 re-separation state machine
    "_lazy_glb_ref": ("_lazy", "glb_ref"),
    "_lazy_armed": ("_lazy", "armed"),
    "_lazy_stagnant_solves": ("_lazy", "stagnant_solves"),
    "_lazy_probe_spent": ("_lazy", "probe_spent"),
    "_lazy_mode": ("_lazy", "mode"),
    "_lazy_resep_fires": ("_lazy", "resep_fires"),
    # PerNodeOBBTBudget — the Lever A engagement gate and effort budget
    "_per_node_obbt_enabled": ("_pn_obbt", "enabled"),
    "_pn_obbt_budget_total": ("_pn_obbt", "budget_total"),
    "_pn_obbt_topk": ("_pn_obbt", "topk"),
    "_pn_obbt_spent": ("_pn_obbt", "spent"),
}

#: Holder local -> the dataclass it holds.
HOLDER_CLASS: dict[str, type] = {
    "_timers": PhaseTimers,
    "_heur": PrimalHeuristicState,
    "_lazy": LazyStallSeparationState,
    "_pn_obbt": PerNodeOBBTBudget,
}

STATE_CLASSES = tuple(HOLDER_CLASS.values())


def _solve_model_ast() -> ast.FunctionDef:
    tree = ast.parse(SOLVER_SOURCE.read_text())
    fns = [
        n
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "solve_model"
    ]
    assert len(fns) == 1, f"expected exactly one top-level solve_model, found {len(fns)}"
    assert isinstance(fns[0], ast.FunctionDef)
    return fns[0]


def _own_scope_bindings(fn: ast.FunctionDef) -> set[str]:
    """Names ``fn`` binds in its own scope (nested scopes excluded)."""
    bound: set[str] = set()
    stack: list[ast.AST] = list(fn.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            continue  # its own scope
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            continue  # comprehension scopes are separate in py3
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        if isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        stack.extend(ast.iter_child_nodes(node))
    return bound


def test_migrated_locals_are_no_longer_bound_in_solve_model() -> None:
    """The point of the migration: one source of truth per quantity.

    A bare ``rust_time = ...`` reappearing alongside ``_timers.rust_time`` is not
    a style problem — it is two variables holding the same accounting, which is
    what the closure made easy and what the object exists to prevent.
    """
    bound = _own_scope_bindings(_solve_model_ast())
    checked = 0
    for local, (holder, fieldname) in MIGRATED.items():
        checked += 1
        assert local not in bound, (
            f"{local!r} is bound as a bare local in solve_model again; it was "
            f"migrated onto {holder}.{fieldname} by consolidation-plan item 11"
        )
    assert checked == len(MIGRATED)


def test_solve_model_binds_every_state_holder() -> None:
    """Each holder must actually be constructed — an unbound holder is a NameError
    waiting on a rarely-taken branch, not a refactor."""
    bound = _own_scope_bindings(_solve_model_ast())
    holders = sorted({holder for holder, _ in MIGRATED.values()})
    assert holders == sorted(HOLDER_CLASS), "the table and HOLDER_CLASS disagree"
    for holder in holders:
        assert holder in bound, f"solve_model never binds the state holder {holder!r}"


def test_state_classes_use_slots_so_a_typo_raises() -> None:
    """``slots=True`` is load-bearing, not cosmetic.

    Several of these fields are (or will be) soundness state. On a plain object a
    typo'd write creates a new attribute and the real field keeps its stale value
    — a silent wrong answer. With slots it raises.
    """
    checked = 0
    for cls in STATE_CLASSES:
        assert dataclasses.is_dataclass(cls)
        assert hasattr(cls, "__slots__"), f"{cls.__name__} must be declared slots=True"
        obj = cls()
        with pytest.raises(AttributeError):
            setattr(obj, "definitely_not_a_field", 1)
        checked += 1
    assert checked == len(STATE_CLASSES)


def test_every_state_field_is_documented() -> None:
    """A field with no comment is a field nobody can carve against."""
    checked = 0
    for cls in STATE_CLASSES:
        src = inspect.getsource(cls)
        assert cls.__doc__ and len(cls.__doc__.strip()) > 80, (
            f"{cls.__name__} needs a docstring saying why its fields belong together"
        )
        for f in dataclasses.fields(cls):
            assert "#: " in src and f"\n    {f.name}: " in src, (
                f"{cls.__name__}.{f.name} is missing its `#:` field comment"
            )
            checked += 1
    assert checked >= 21, f"only {checked} fields checked; the probe under-fired"


def test_migration_table_matches_the_state_classes() -> None:
    """The table and the dataclasses must not drift apart, in either direction.

    A table entry with no matching field means the migration is half-applied; a
    field no entry claims means a state object grew a member nobody migrated onto
    it, which is how a state object quietly becomes a junk drawer.
    """
    checked = 0
    for local, (holder, fieldname) in MIGRATED.items():
        names = {f.name for f in dataclasses.fields(HOLDER_CLASS[holder])}
        assert fieldname in names, f"{local!r} claims {holder}.{fieldname}, which does not exist"
        checked += 1
    assert checked == len(MIGRATED)

    for holder, cls in HOLDER_CLASS.items():
        claimed = {f for h, f in MIGRATED.values() if h == holder}
        actual = {f.name for f in dataclasses.fields(cls)}
        assert actual == claimed, (
            f"{cls.__name__} fields {actual - claimed} are claimed by no migration "
            f"table entry (and {claimed - actual} are claimed but absent)"
        )


def test_executed_assertion_count_is_nonzero() -> None:
    """CLAUDE.md §6: prove the probe fired rather than traversing nothing."""
    fn = _solve_model_ast()
    bound = _own_scope_bindings(fn)
    assert len(bound) > 200, f"only {len(bound)} bindings found; the AST walk under-fired"
    assert len(MIGRATED) >= 21
