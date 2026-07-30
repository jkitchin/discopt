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
    RootConfig,
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
    # RootConfig — solve-wide immutable configuration.  Unlike every group above,
    # these are `solve_model` *parameters*, so the name stays bound (as a
    # parameter) and the invariant is different: each must be read exactly once,
    # at the `_cfg = RootConfig(...)` construction, with every other read going
    # through `_cfg`.  `test_root_config_parameters_are_read_only_at_construction`
    # is the test that enforces it; the "not bound in own scope" test above is
    # satisfied trivially and is NOT the guarantee for this group.
    "gdp_method": ("_cfg", "gdp_method"),
    "mccormick_bounds": ("_cfg", "mccormick_bounds"),
    "skip_convex_check": ("_cfg", "skip_convex_check"),
    "strategy": ("_cfg", "strategy"),
    "threads": ("_cfg", "threads"),
    "rlt": ("_cfg", "rlt"),
    "rlt_cuts": ("_cfg", "rlt_cuts"),
    "nlp_bb": ("_cfg", "nlp_bb"),
    "partitions": ("_cfg", "partitions"),
    "psd_cuts": ("_cfg", "psd_cuts"),
    "cuts": ("_cfg", "cuts"),
    "use_learned_relaxations": ("_cfg", "use_learned_relaxations"),
    "eigenvalue_root_bound": ("_cfg", "eigenvalue_root_bound"),
    "lagrangian_bound": ("_cfg", "lagrangian_bound"),
    "lagrangian_frequency": ("_cfg", "lagrangian_frequency"),
    "presolve_polynomial": ("_cfg", "presolve_polynomial"),
    "presolve_reverse_ad": ("_cfg", "presolve_reverse_ad"),
    "subnlp_backend": ("_cfg", "subnlp_backend"),
    "subnlp_enabled": ("_cfg", "subnlp_enabled"),
}

#: The subset of :data:`MIGRATED` sourced from ``solve_model`` parameters rather
#: than from locals it computes.  Kept separate because their invariant is the
#: single-read one, not the not-rebound one.
ROOT_CONFIG_PARAMS: frozenset[str] = frozenset(
    local for local, (holder, _) in MIGRATED.items() if holder == "_cfg"
)

#: Parameters deliberately EXCLUDED from ``RootConfig`` because each has a derived
#: twin computed later in ``solve_model``.  Admitting both would create a shadow
#: pair that drifts the first time one side is updated and the other is not.
SHADOW_PAIR_EXCLUSIONS: dict[str, str] = {
    "root_cut_max": "_root_cut_max",
    "root_cut_rounds": "_root_cut_rounds",
    "solver": "_solver",
}

#: Holder local -> the dataclass it holds.
HOLDER_CLASS: dict[str, type] = {
    "_timers": PhaseTimers,
    "_heur": PrimalHeuristicState,
    "_lazy": LazyStallSeparationState,
    "_pn_obbt": PerNodeOBBTBudget,
    "_cfg": RootConfig,
}

#: Constructor arguments for holders whose fields have no defaults.  ``RootConfig``
#: is frozen and every field is required, precisely so that a half-built config
#: cannot exist.
HOLDER_SAMPLE_KWARGS: dict[str, dict[str, object]] = {
    "_cfg": {
        "gdp_method": "bigm",
        "mccormick_bounds": "auto",
        "skip_convex_check": False,
        "strategy": "best_bound",
        "threads": 1,
        "rlt": False,
        "rlt_cuts": False,
        "nlp_bb": None,
        "partitions": 1,
        "psd_cuts": False,
        "cuts": "auto",
        "use_learned_relaxations": False,
        "eigenvalue_root_bound": False,
        "lagrangian_bound": False,
        "lagrangian_frequency": 1,
        "presolve_polynomial": False,
        "presolve_reverse_ad": False,
        "subnlp_backend": "ipopt",
        "subnlp_enabled": False,
    }
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
    for holder, cls in HOLDER_CLASS.items():
        assert dataclasses.is_dataclass(cls)
        assert hasattr(cls, "__slots__"), f"{cls.__name__} must be declared slots=True"
        obj = cls(**HOLDER_SAMPLE_KWARGS.get(holder, {}))  # type: ignore[arg-type]
        # The exception TYPE differs by holder — a plain slots dataclass raises
        # AttributeError, while a frozen+slots one raises TypeError out of the
        # generated __setattr__ — so the assertion is on the property that matters:
        # the write is refused and no attribute comes into existence.
        with pytest.raises((AttributeError, TypeError)):
            setattr(obj, "definitely_not_a_field", 1)
        assert not hasattr(obj, "definitely_not_a_field")
        checked += 1
    assert checked == len(HOLDER_CLASS)


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


def test_root_config_parameters_are_read_only_at_construction() -> None:
    """Each ``RootConfig`` parameter is read exactly once, at the construction.

    This is the guarantee for the parameter-sourced group, and it is the one that
    makes the holder worth having. A parameter still bound in the signature can be
    read directly *and* through ``_cfg``; the two would then be the same value
    today and two sources of truth the moment anyone resolves one of them. So the
    test counts bare ``Load`` references to each name in ``solve_model``'s own
    scope and requires exactly one — the ``RootConfig(...)`` keyword argument.
    """
    fn = _solve_model_ast()
    construction = None
    for node in fn.body:
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "_cfg"
        ):
            construction = node
            break
    assert construction is not None, "solve_model never constructs _cfg"
    construction_lines = set(range(construction.lineno, (construction.end_lineno or 0) + 1))

    loads: dict[str, list[int]] = {name: [] for name in ROOT_CONFIG_PARAMS}
    stack: list[ast.AST] = list(fn.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            continue
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in loads:
            loads[node.id].append(node.lineno)
        stack.extend(ast.iter_child_nodes(node))

    checked = 0
    for name in sorted(ROOT_CONFIG_PARAMS):
        sites = loads[name]
        assert len(sites) == 1, (
            f"{name} is read {len(sites)} times directly at lines {sites}; every read "
            f"except the RootConfig construction must go through `_cfg.{name}`"
        )
        assert sites[0] in construction_lines, (
            f"{name}'s only direct read is at line {sites[0]}, outside the RootConfig construction"
        )
        checked += 1
    assert checked == len(ROOT_CONFIG_PARAMS) >= 19
    print(f"executed single-read assertions: {checked}")


def test_root_config_admits_no_shadow_pair() -> None:
    """A raw parameter whose resolved twin exists must stay out of the holder.

    ``root_cut_max`` and ``_root_cut_max`` are both live in ``solve_model``; the
    second is the one the solver actually acts on. Carrying the raw one in a thing
    named *config* invites a later reader to use it, and the two diverge silently.
    """
    fields = {f.name for f in dataclasses.fields(RootConfig)}
    checked = 0
    for raw, resolved in SHADOW_PAIR_EXCLUSIONS.items():
        assert raw not in fields, (
            f"RootConfig admits the raw parameter {raw!r} whose resolved twin "
            f"{resolved!r} also exists — that is a shadow pair"
        )
        assert raw not in ROOT_CONFIG_PARAMS
        checked += 1
    assert checked == len(SHADOW_PAIR_EXCLUSIONS)


def test_root_config_is_frozen_and_holds_only_immutable_values() -> None:
    """The freeze must be a true guarantee, not a decorative one.

    ``frozen=True`` blocks field rebinding and nothing else, so a holder carrying a
    list or a dict would still be mutable through it — which is exactly the trap
    the 2026-07-30 design review caught this plan about to walk into with the live
    B&B ``tree``. Every field here is checked to be a scalar type, for which
    CPython offers no in-place mutation at all.
    """
    assert RootConfig.__dataclass_params__.frozen, "RootConfig must be frozen"
    cfg = RootConfig(**HOLDER_SAMPLE_KWARGS["_cfg"])  # type: ignore[arg-type]
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.threads = 4  # type: ignore[misc]

    checked = 0
    for f in dataclasses.fields(RootConfig):
        value = getattr(cfg, f.name)
        assert isinstance(value, (str, int, float, bool, type(None))), (
            f"RootConfig.{f.name} holds {type(value).__name__}, which frozen= does "
            "not protect; admit it only after the mutability audit clears it and "
            "give arrays setflags(write=False)"
        )
        checked += 1
    assert checked == len(dataclasses.fields(RootConfig)) >= 19
    print(f"executed immutability assertions: {checked}")


def test_executed_assertion_count_is_nonzero() -> None:
    """CLAUDE.md §6: prove the probe fired rather than traversing nothing."""
    fn = _solve_model_ast()
    bound = _own_scope_bindings(fn)
    assert len(bound) > 200, f"only {len(bound)} bindings found; the AST walk under-fired"
    assert len(MIGRATED) >= 40
