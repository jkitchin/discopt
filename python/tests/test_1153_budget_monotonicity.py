"""#1153: a bigger ``time_limit`` must not buy a worse answer.

The defect
----------

On ``nvs19`` a 30 s budget reached ``-1098.2`` over 38 403 nodes and a 60 s
budget reached ``-1001.2`` over 7 619: **doubling the budget made the answer
worse and explored 5x fewer nodes**. No false certificate was involved — the
reported incumbents were feasible and the status was not ``optimal`` — so this
is a completeness miss plus a *monotonicity* defect, and the monotonicity half
is the general one. A user who doubles ``time_limit`` and gets a worse answer
has no way to reason about the solver.

The mechanism is #1116's role split read one step further. Role 1 (*"when do we
stop?"*) is the caller's ``time_limit``, and reading a clock for it is correct by
definition. Role 2 (*"how much work does this stage do?"*) is a sub-budget carved
as a fraction of it. Carving role 2 out of role 1 is not itself wrong — a stage
must not outlive the solve — but it becomes wrong when the carve never
**saturates**: a root stage whose grant keeps growing with the caller's budget
goes on separating cuts, every subsequent node LP carries them, the per-node cost
therefore rises with ``time_limit``, and the tree the remaining budget can cover
shrinks. That is the coupling #1116 predicted and #1153 measured the harm of.

What this file pins
-------------------

``test_carve_saturates`` / ``test_carve_is_inert_when_flag_off``
    The unit contract of :func:`discopt.solver_tuning.saturate_role2`.

``test_no_unsaturated_role2_carve``
    The **ratchet**, in the idiom of ``test_912_wall_budget_inventory``: every
    sub-budget in the package computed by multiplying or dividing a role-1 budget
    is either provably bounded above by a constant, or recorded in :data:`KNOWN`
    with the reason it is not a role-2 carve. A new uncapped carve fails here.

``test_incumbent_quality_is_monotone_in_time_limit``
    The behavioural gate: over a panel of instances and an increasing ladder of
    budgets, the incumbent never gets worse. This is the property #1153's
    "definition of done" names as *the* gate — an implementation that solves
    ``nvs19`` without restoring it has fixed an instance, not the class.

Every probe here reports an executed-assertion count and fails when it is zero
(CLAUDE.md §6): a scanner that matches nothing, or a panel where no instance
produced two comparable incumbents, reads as a pass otherwise.
"""

from __future__ import annotations

import ast
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

from pathlib import Path

import pytest
from discopt import solver_tuning
from discopt.modeling.core import from_nl
from discopt.solver_tuning import (
    HEURISTIC_ENTRY_SHARE,
    ROLE2_SATURATION_S,
    SolverTuning,
    heuristic_entry_share,
    saturate_role2,
)

_PKG = Path(__file__).resolve().parents[1] / "discopt"
_DATA = Path(__file__).resolve().parent / "data" / "minlplib_nl"


# --------------------------------------------------------------------------- #
# The carve helper                                                            #
# --------------------------------------------------------------------------- #


def _with_saturation(on: bool):
    return solver_tuning.enter_scope(SolverTuning(budget_saturation=on))


@pytest.mark.unit
def test_carve_saturates():
    """With the flag on, a carve stops at the value it takes at the reference."""
    token = _with_saturation(True)
    try:
        # Below the reference budget the carve is untouched...
        assert saturate_role2(0.25 * 60.0, 0.25) == pytest.approx(15.0)
        # ...at it, it is exactly the ceiling...
        assert saturate_role2(0.25 * ROLE2_SATURATION_S, 0.25) == pytest.approx(
            0.25 * ROLE2_SATURATION_S
        )
        # ...and beyond it, it does not grow.
        assert saturate_role2(0.25 * 10_000.0, 0.25) == pytest.approx(0.25 * ROLE2_SATURATION_S)
        assert saturate_role2(0.1 * 10_000.0, 0.1) == pytest.approx(0.1 * ROLE2_SATURATION_S)
    finally:
        solver_tuning.reset_current(token)


@pytest.mark.unit
def test_carve_is_inert_when_flag_off():
    """Flag off must be the legacy path, byte for byte."""
    token = _with_saturation(False)
    try:
        for seconds in (0.5, 15.0, 2_500.0):
            assert saturate_role2(seconds, 0.25) == pytest.approx(seconds)
    finally:
        solver_tuning.reset_current(token)


# --------------------------------------------------------------------------- #
# The finder-entry share                                                      #
# --------------------------------------------------------------------------- #


def _with_entry_share(on: bool):
    return solver_tuning.enter_scope(SolverTuning(heuristic_entry_share=on))


@pytest.mark.unit
def test_finder_entry_share_is_bounded_when_on():
    """A finder heuristic must fit a SHARE of the remainder, not all of it."""
    token = _with_entry_share(True)
    try:
        assert heuristic_entry_share() == pytest.approx(HEURISTIC_ENTRY_SHARE)
        assert 0.0 < HEURISTIC_ENTRY_SHARE < 1.0
    finally:
        solver_tuning.reset_current(token)


@pytest.mark.unit
def test_finder_entry_share_is_the_legacy_rule_when_off():
    """Flag off must reproduce "it may consume everything" exactly."""
    token = _with_entry_share(False)
    try:
        assert heuristic_entry_share() == 1.0
    finally:
        solver_tuning.reset_current(token)


# --------------------------------------------------------------------------- #
# The static ratchet                                                          #
# --------------------------------------------------------------------------- #

#: Names that carry a role-1 (caller-owned) wall budget.
_ROLE1_NAMES = frozenset(
    {
        "time_limit",
        "total_time_limit",
        "time_budget",
        "remaining",
        "_remaining",
        "_remaining_tl",
        "_outer_budget",
        "_node_remaining",
        "_root_remaining",
        "_probe_remaining",
        "_ms_remaining",
        "_hg_remaining",
    }
)

#: Calls that saturate their argument (#1153).
_SATURATING = frozenset({"saturate_role2", "_role2_saturate"})

#: Calls that return their argument unchanged (or a no-clock value); a carve
#: wrapped in one of these is bounded exactly when the carve inside it is.
_PASSTHROUGH = frozenset({"_role2_budget", "_role2_deadline", "_role2_horizon"})


def _bounded(node: ast.AST) -> bool:
    """Is this expression bounded above by a constant?

    ``min`` needs one bounded argument, ``max`` needs all of them, an ALL-CAPS
    name is a module constant, and arithmetic is bounded when both sides are.
    Everything else — a call whose value this cannot see, an attribute, a plain
    lower-case name — is treated as unbounded, so the ratchet errs toward
    *reporting* a site rather than toward silence.
    """
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (int, float)) and not isinstance(node.value, bool)
    if isinstance(node, ast.Name):
        return node.id.lstrip("_").isupper()
    if isinstance(node, ast.Call):
        func = node.func
        name = (
            func.id
            if isinstance(func, ast.Name)
            else (func.attr if isinstance(func, ast.Attribute) else "")
        )
        if name in _SATURATING:
            return True
        if name in _PASSTHROUGH:
            return bool(node.args) and _bounded(node.args[0])
        if name == "min":
            return any(_bounded(a) for a in node.args)
        if name == "max":
            return bool(node.args) and all(_bounded(a) for a in node.args)
        if name in ("float", "abs"):
            return all(_bounded(a) for a in node.args)
        return False
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mult, ast.Div, ast.Add, ast.Sub)):
        return _bounded(node.left) and _bounded(node.right)
    if isinstance(node, ast.IfExp):
        return _bounded(node.body) and _bounded(node.orelse)
    return False


def _scan() -> set[tuple[str, str]]:
    """Every carve of a role-1 budget that is NOT bounded above by a constant."""
    found: set[tuple[str, str]] = set()
    for root, dirs, files in os.walk(_PKG):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for f in sorted(files):
            if not f.endswith(".py"):
                continue
            path = Path(root) / f
            src = path.read_text().splitlines()
            # Deliberately unguarded: a file that will not parse must fail loudly
            # rather than be skipped into an "all clear" (CLAUDE.md §7).
            tree = ast.parse("\n".join(src), filename=str(path))
            parents: dict[ast.AST, ast.AST] = {}
            for node in ast.walk(tree):
                for child in ast.iter_child_nodes(node):
                    parents[child] = node
            seen: set[int] = set()
            for node in ast.walk(tree):
                if not isinstance(node, ast.BinOp) or not isinstance(node.op, (ast.Mult, ast.Div)):
                    continue
                names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
                if not names & _ROLE1_NAMES:
                    continue
                # Widen to the whole value expression: ``0.25 * time_limit`` is
                # bounded when it sits inside ``min(..., 30.0)``, and judging the
                # product alone would report every capped site in the package.
                top: ast.AST = node
                while (
                    top in parents
                    and isinstance(parents[top], ast.expr)
                    and not isinstance(parents[top], (ast.Compare, ast.BoolOp))
                ):
                    top = parents[top]
                if top.lineno in seen:
                    continue
                seen.add(top.lineno)
                if _bounded(top):
                    continue
                found.add((str(path.relative_to(_PKG)), src[top.lineno - 1].strip()))
    return found


# (module-relative path, source line, why it is not an unsaturated role-2 carve).
#
# ``split``   the value is a division of the caller's own role-1 budget between
#             two consumers, not a work allowance — growing with ``time_limit``
#             is the whole point.
# ``derived`` the multiplicand is itself already bounded (by a constant, by a
#             caller, or by a cap on the very next line), so the product is too;
#             the scanner cannot see across that boundary.
# ``counted`` the wall figure is a secondary bound; a deterministic count is the
#             real one.
KNOWN: tuple[tuple[str, str, str], ...] = (
    (
        "_relax/root_reduce.py",
        "obbt_budget = remaining * float(obbt_stage_frac)",
        "derived",  # ``remaining`` descends from the (saturated) root-fixpoint grant
    ),
    (
        "modeling/core.py",
        "_fb_reserve = 0.35 * _remaining_tl",
        "split",  # the #844 fallback's share of the caller's budget
    ),
    (
        "solver.py",
        "_bound_reserve = min(float(rr_reserve_s), 0.5 * remaining)",
        "derived",  # ``rr_reserve_s`` <= ``_ROOT_FALLBACK_RESERVE_S`` (3 s)
    ),
    (
        "solver.py",
        "per_solve_limit = max(0.05, time_limit / (2 * len(candidate_var_indices) + 1))",
        "split",  # divides the helper's own budget across its candidate variables
    ),
    (
        "solver.py",
        "_route_budget = max(",
        "split",  # the auto-routed sub-solver's own role-1 budget
    ),
    (
        "solver.py",
        "deadline=min(",
        "derived",  # an absolute deadline; the duration inside ceilings at 15 s
    ),
    (
        "solver.py",
        "self.budget = (_ROUND_TIME_FRAC if frac is None else float(frac)) * float(time_limit)",
        "counted",  # ``_ROUND_ATTEMPT_CAP`` (64) is the real bound
    ),
    (
        "solver.py",
        "float(time_limit)",
        "derived",  # the ``deterministic`` no-clock arm; the live arm caps at
        # ``_SIMPLEX_MILP_BUDGET_CAP_S``
    ),
    (
        "solvers/amp.py",
        "iter_budget = remaining / horizon",
        "derived",  # capped at 60 s by the ``return min(...)`` on the next line
    ),
    (
        "solvers/oa.py",
        "budget = remaining * _MASTER_NO_INCUMBENT_BUDGET_FRAC",
        "split",  # the OA master's share of the caller's budget
    ),
)

_KNOWN_KEYS = {(p, s) for p, s, _ in KNOWN}


@pytest.mark.unit
def test_no_unsaturated_role2_carve():
    """A role-2 sub-budget must saturate, or be recorded as not being one."""
    found = _scan()
    assert found, "the scanner matched nothing — it has stopped working (CLAUDE.md §6)"
    new = sorted(found - _KNOWN_KEYS)
    assert not new, (
        "unsaturated role-2 wall carve(s) — #1153.\n"
        "A sub-budget carved as a fraction of the caller's ``time_limit`` must\n"
        "stop growing at some point, or a bigger budget buys more preprocessing\n"
        "instead of more search and the answer gets WORSE (nvs19: 30 s ->\n"
        "-1098.2 over 38403 nodes, 60 s -> -1001.2 over 7619). Wrap it in\n"
        "``solver_tuning.saturate_role2(seconds, frac)``, give it a constant\n"
        "``min(...)`` ceiling, or record it in KNOWN with a category.\n\n"
        + "\n".join(f"  {p}: {s}" for p, s in new)
    )


@pytest.mark.unit
def test_recorded_carves_still_exist():
    """The ratchet must not rot into stale bookkeeping."""
    stale = sorted(_KNOWN_KEYS - _scan())
    assert not stale, "KNOWN lists carve(s) that no longer exist — remove them:\n" + "\n".join(
        f"  {p}: {s}" for p, s in stale
    )


@pytest.mark.unit
def test_the_saturated_sites_stay_saturated():
    """The carves #1153 saturated must keep their wrapper.

    Dropping one is invisible to the ratchet above: the line text would change,
    the author would re-record it in KNOWN, and both tests would pass while the
    coupling came back. So the count is pinned here.
    """
    sites = []
    for path in sorted(_PKG.rglob("*.py")):
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith(("def ", "#", "*", "from ", "import ")):
                continue
            if "saturate_role2(" in stripped or "_role2_saturate(" in stripped:
                sites.append(f"{path.relative_to(_PKG)}:{lineno}")
    # The six carves #1153 saturated: the root LP probe, the two root cut-pool
    # separations, the cumulative per-node OBBT grant, the root fixpoint, and the
    # LP-spatial engine's root OBBT.
    assert len(sites) == 6, (
        f"the #1153 saturation count changed ({len(sites)} call sites, expected 6). "
        "Lowering it means a carve went back to tracking the caller's budget.\n"
        + "\n".join(f"  {s}" for s in sites)
    )


# --------------------------------------------------------------------------- #
# The behavioural gate                                                        #
# --------------------------------------------------------------------------- #

#: The budgets the throughput collapse was measured at. 5 s refuses the
#: feasibility pump and 10 s admits it; see the plan doc §6.2.
_SMALL, _LARGE = 5.0, 10.0

#: Instances where the collapse reproduced with ZERO spread over three
#: repetitions (7 -> 3 and 7 -> 3 nodes). Selected by that measured property, not
#: by name (CLAUDE.md §2) — the selection run is
#: ``scratchpad/i1153/reps.py``. ``tspn12`` is deliberately excluded: the pump is
#: PRODUCTIVE there (it finds 262.647 against 282.244), so it is the case the
#: share rule should *not* be judged on, and §6.3 records that it loses that
#: incumbent.
_COLLAPSE_PANEL = ("heatexch_gen2", "tspn10")

#: A ladder for the general property, over the budgets where #1153's harm lives.
_LADDER = (5.0, 10.0, 20.0)


def _nodes(path: Path, tl: float) -> int:
    return int(from_nl(str(path)).solve(time_limit=tl, gap_tolerance=1e-4).node_count or 0)


@pytest.mark.slow
@pytest.mark.correctness
@pytest.mark.skipif(not _DATA.is_dir(), reason="MINLPLib corpus not vendored")
def test_finder_entry_share_stops_the_throughput_collapse():
    """Doubling the budget must not HALVE the tree (#1153's throughput half).

    The legacy arm is run as a **control**, not for symmetry: without it, an arm
    that collapsed for some unrelated reason — or a panel where the pump stopped
    being admitted at all — would read as a pass (CLAUDE.md §6). The control must
    reproduce the collapse for the treatment arm's result to mean anything.
    """
    control_collapsed = 0
    failures: list[str] = []
    compared = 0
    for name in _COLLAPSE_PANEL:
        path = _DATA / f"{name}.nl"
        if not path.exists():
            continue
        token = _with_entry_share(False)
        try:
            off_small, off_large = _nodes(path, _SMALL), _nodes(path, _LARGE)
        finally:
            solver_tuning.reset_current(token)
        token = _with_entry_share(True)
        try:
            on_small, on_large = _nodes(path, _SMALL), _nodes(path, _LARGE)
        finally:
            solver_tuning.reset_current(token)
        compared += 1
        if off_large < off_small:
            control_collapsed += 1
        if on_large < on_small:
            failures.append(
                f"{name}: share ON still collapses {on_small} -> {on_large} nodes "
                f"({_SMALL}s -> {_LARGE}s); control {off_small} -> {off_large}"
            )

    assert compared > 0, "no panel instance was vendored — this test measured nothing"
    assert control_collapsed > 0, (
        "the legacy arm did not reproduce the collapse on any instance, so the "
        "treatment arm's result is unfalsifiable here (CLAUDE.md §6) — re-derive "
        "the panel with scratchpad/i1153/reps.py"
    )
    assert not failures, "\n".join(failures)


@pytest.mark.slow
@pytest.mark.correctness
@pytest.mark.skipif(not _DATA.is_dir(), reason="MINLPLib corpus not vendored")
def test_incumbent_quality_is_monotone_in_time_limit():
    """More budget must never yield a worse incumbent (#1153's gate)."""
    token = _with_entry_share(True)
    comparisons = 0
    violations: list[str] = []
    try:
        for name in _COLLAPSE_PANEL + ("tspn12",):
            path = _DATA / f"{name}.nl"
            if not path.exists():
                continue
            rung = [
                (tl, from_nl(str(path)).solve(time_limit=tl, gap_tolerance=1e-4).objective)
                for tl in _LADDER
            ]
            for (tl_a, obj_a), (tl_b, obj_b) in zip(rung, rung[1:]):
                if obj_a is None:
                    continue  # nothing to be monotone about yet
                comparisons += 1
                if obj_b is None:
                    violations.append(f"{name}: {tl_a}s -> {obj_a}, {tl_b}s -> no incumbent")
                elif obj_b > obj_a + 1e-6 * max(1.0, abs(obj_a)):
                    violations.append(f"{name}: {tl_a}s -> {obj_a}, {tl_b}s -> {obj_b}")
    finally:
        solver_tuning.reset_current(token)

    assert comparisons > 0, (
        "no instance produced an incumbent at two budgets — this test measured "
        "nothing (CLAUDE.md §6)"
    )
    assert not violations, (
        f"incumbent quality fell as the budget grew, over {comparisons} comparison(s):\n"
        + "\n".join(f"  {v}" for v in violations)
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
