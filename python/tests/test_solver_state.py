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
from discopt.solver.state import PhaseTimers, PrimalHeuristicState

pytestmark = pytest.mark.smoke

SOLVER_SOURCE = Path(S.__file__)

#: Every local that has been migrated onto a state object, mapped to the holder
#: it now lives on. Extend this table when a further group is threaded; the tests
#: below then enforce the new entries automatically.
MIGRATED: dict[str, str] = {
    "rust_time": "_timers",
    "jax_time": "_timers",
    "t_rust_start": "_timers",
    "t_jax_start": "_timers",
    "_subnlp_backend_fn": "_heur",
    "_subnlp_calls": "_heur",
    "_subnlp_feasible": "_heur",
    "_subnlp_incumbent_updates": "_heur",
    "_lns_lb_calls": "_heur",
    "_lns_dive_calls": "_heur",
    "_lns_swap_misses": "_heur",
}

STATE_CLASSES = (PhaseTimers, PrimalHeuristicState)


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
    for local in MIGRATED:
        checked += 1
        assert local not in bound, (
            f"{local!r} is bound as a bare local in solve_model again; it was "
            f"migrated onto {MIGRATED[local]!r} by consolidation-plan item 11"
        )
    assert checked == len(MIGRATED)


def test_solve_model_binds_every_state_holder() -> None:
    """Each holder must actually be constructed — an unbound holder is a NameError
    waiting on a rarely-taken branch, not a refactor."""
    bound = _own_scope_bindings(_solve_model_ast())
    holders = sorted(set(MIGRATED.values()))
    assert holders, "the migration table is empty"
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
    assert checked >= 11, f"only {checked} fields checked; the probe under-fired"


def test_migration_table_matches_the_state_classes() -> None:
    """The table above and the dataclasses must not drift apart.

    Every migrated local maps to a field named by stripping its leading
    underscores — the deliberately mechanical rule from ``solver/state.py`` — so a
    table entry with no matching field means the migration is half-applied.
    """
    fields_by_holder = {
        "_timers": {f.name for f in dataclasses.fields(PhaseTimers)},
        "_heur": {f.name for f in dataclasses.fields(PrimalHeuristicState)},
    }
    assert set(fields_by_holder) == set(MIGRATED.values())
    checked = 0
    for local, holder in MIGRATED.items():
        expected = local.lstrip("_")
        assert expected in fields_by_holder[holder], (
            f"{local!r} should map to {holder}.{expected}, which does not exist"
        )
        checked += 1
    assert checked == len(MIGRATED)
    # every field is accounted for by the table, in both directions
    covered = {local.lstrip("_") for local in MIGRATED}
    for holder, names in fields_by_holder.items():
        assert names <= covered, (
            f"{holder} has fields no migration-table entry claims: {names - covered}"
        )


def test_executed_assertion_count_is_nonzero() -> None:
    """CLAUDE.md §6: prove the probe fired rather than traversing nothing."""
    fn = _solve_model_ast()
    bound = _own_scope_bindings(fn)
    assert len(bound) > 200, f"only {len(bound)} bindings found; the AST walk under-fired"
    assert len(MIGRATED) >= 11
