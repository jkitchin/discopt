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
    The **ratchet**, in the idiom of ``test_912_wall_budget_inventory``: a
    sub-budget computed by multiplying or dividing a role-1 budget is either
    provably bounded above by a constant, or recorded in :data:`KNOWN` with the
    reason it is not a role-2 carve. A new uncapped carve fails here.

    **The coverage limit, stated rather than implied.** "A role-1 budget" means a
    name in :data:`_ROLE1_NAMES`, a 13-name whitelist. A carve off a role-1 value
    held under some *other* name — ``_budget_left * 0.42`` — is invisible to this
    scan, and no amount of AST work fixes that without whole-package dataflow.
    The ratchet is a tripwire on the spellings this codebase actually uses, not a
    proof of absence.

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
import statistics

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

from pathlib import Path

import pytest
from discopt import solver_tuning
from discopt.modeling.core import ObjectiveSense, from_nl
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
        "outer_deadline",
    }
)

#: Calls that RETURN a role-1 remaining budget. A carve off one of these is a
#: role-1 carve even though no whitelisted Name appears — ``0.5 *
#: _remaining_budget()`` is a ``Call``, not a ``Name``, and the allowlist alone
#: never sees it.
_ROLE1_CALLS = frozenset({"_remaining_budget", "_fb_left", "_remaining", "_remaining_budget_s"})

#: Clock reads. ``deadline - time.perf_counter()`` is a role-1 remaining budget
#: spelled as a subtraction, which is how the helper this PR itself deleted
#: (``_heur_stage_deadline``) carved ``share * (_deadline - _now)`` straight past
#: the first version of this scan.
_CLOCK_FUNCS = frozenset({"perf_counter", "monotonic", "_now"})

#: Calls that saturate their argument (#1153).
_SATURATING = frozenset({"saturate_role2", "_role2_saturate"})

#: Calls that return their argument unchanged (or a no-clock value); a carve
#: wrapped in one of these is bounded exactly when the carve inside it is.
#:
#: ``_role2_slice`` joined the family with #1187. It is the ``time_limit``-valued
#: sibling of the other three — a nested ``solve_model``'s budget parameter has no
#: ``None``/``math.inf`` spelling, so its no-clock value is the caller's own
#: ``time_limit``. That makes it *more* bounded than ``_role2_horizon``, which is
#: already listed and whose no-clock value is ``math.inf``; either way the
#: ``deterministic`` regime is not what #1153's saturation rule is about, and the
#: carve to judge is the one inside the wrapper.
_PASSTHROUGH = frozenset({"_role2_budget", "_role2_deadline", "_role2_horizon", "_role2_slice"})


def _is_clock_read(node: ast.AST) -> bool:
    """``time.perf_counter()`` / ``time.monotonic()`` / the ``_now()`` seam."""
    if not isinstance(node, ast.Call):
        return False
    f = node.func
    if isinstance(f, ast.Attribute):
        return f.attr in _CLOCK_FUNCS
    return isinstance(f, ast.Name) and f.id in _CLOCK_FUNCS


def _carries_role1(node: ast.AST) -> bool:
    """Does this expression carry a role-1 (caller-owned) remaining budget?

    Three spellings, all of which occur in this package and only the first of
    which a name allowlist can see:

    * a whitelisted **name** — ``time_limit``, ``remaining``, …;
    * a **call** that returns one — ``_remaining_budget()``;
    * a **subtraction** of a clock read from a deadline — ``_deadline - _now``,
      ``deadline - time.perf_counter()``.

    The third is not hypothetical: the helper this PR deleted
    (``_heur_stage_deadline``) carved ``share * (_deadline - _now)``, an
    unsaturated role-2 carve, and the first version of this scan reported it
    clean. The ratchet must see the defect its own PR introduced.
    """
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and n.id in _ROLE1_NAMES:
            return True
        if isinstance(n, ast.Call):
            f = n.func
            name = (
                f.id
                if isinstance(f, ast.Name)
                else (f.attr if isinstance(f, ast.Attribute) else "")
            )
            if name in _ROLE1_CALLS:
                return True
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Sub):
            if _is_clock_read(n.right) or (
                isinstance(n.right, ast.Name) and "deadline" in n.right.id.lower()
            ):
                return True
            if isinstance(n.left, ast.Name) and "deadline" in n.left.id.lower():
                return True
    return False


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


def _scan() -> dict[tuple[str, str], int]:
    """Every carve of a role-1 budget that is NOT bounded above by a constant,
    **counted per key**.

    A set keyed on ``(path, first source line)`` is not enough, and the gap is the
    one this ratchet exists to close. Several recorded lines are generic prefixes
    of a widened multi-line expression — ``deadline=min(``, ``_route_budget =
    max(``, ``float(time_limit)`` — so a NEW unbounded carve whose widened
    expression happens to start with the same text is absorbed by the existing
    entry and reported clean, while ``test_recorded_carves_still_exist`` also
    stays green because the recorded line is still there. Verified by injecting
    such a carve into ``solve_model``: the set-based scan reported nothing.

    Counting closes it: a second site landing on a recorded line takes the count
    from 1 to 2 and fails against the pinned count in :data:`KNOWN`.
    """
    found: dict[tuple[str, str], int] = {}
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
                if not _carries_role1(node):
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
                key = (str(path.relative_to(_PKG)), src[top.lineno - 1].strip())
                found[key] = found.get(key, 0) + 1
    return found


# (module-relative path, source line, count, why it is not an unsaturated role-2
# carve). The COUNT is load-bearing: see ``_scan``. Every entry is 1 today; a new
# carve colliding with one of the generic recorded prefixes would make it 2 and
# fail, which a set-keyed ratchet missed entirely.
#
# ``split``   the value is a division of the caller's own role-1 budget between
#             two consumers, not a work allowance — growing with ``time_limit``
#             is the whole point.
# ``derived`` the multiplicand is itself already bounded (by a constant, by a
#             caller, or by a cap on the very next line), so the product is too;
#             the scanner cannot see across that boundary.
# ``counted`` the wall figure is a secondary bound; a deterministic count is the
#             real one.
KNOWN: tuple[tuple[str, str, int, str], ...] = (
    (
        "_relax/root_reduce.py",
        "obbt_budget = remaining * float(obbt_stage_frac)",
        1,
        "derived",  # ``remaining`` descends from the (saturated) root-fixpoint grant
    ),
    (
        "modeling/core.py",
        "_fb_reserve = 0.35 * _remaining_tl",
        1,
        "split",  # the #844 fallback's share of the caller's budget
    ),
    (
        "solver.py",
        "_bound_reserve = min(float(rr_reserve_s), 0.5 * remaining)",
        1,
        "derived",  # ``rr_reserve_s`` <= ``_ROOT_FALLBACK_RESERVE_S`` (3 s)
    ),
    (
        "solver.py",
        "per_solve_limit = max(0.05, time_limit / (2 * len(candidate_var_indices) + 1))",
        1,
        "split",  # divides the helper's own budget across its candidate variables
    ),
    (
        "solver.py",
        "_route_budget = max(",
        1,
        "split",  # the auto-routed sub-solver's own role-1 budget
    ),
    (
        "solver.py",
        "deadline=min(",
        1,
        "derived",  # an absolute deadline; the duration inside ceilings at 15 s
    ),
    (
        "solver.py",
        "self.budget = (_ROUND_TIME_FRAC if frac is None else float(frac)) * float(time_limit)",
        1,
        "counted",  # ``_ROUND_ATTEMPT_CAP`` (64) is the real bound
    ),
    (
        "solver.py",
        "float(time_limit)",
        1,
        "derived",  # the ``deterministic`` no-clock arm; the live arm caps at
        # ``_SIMPLEX_MILP_BUDGET_CAP_S``
    ),
    (
        "solvers/amp.py",
        "iter_budget = remaining / horizon",
        1,
        "derived",  # capped at 60 s by the ``return min(...)`` on the next line
    ),
    (
        "solver.py",
        "return min(",
        1,
        # ``_finder_stage_deadline``: the #1153 stage cap itself. A share of the
        # caller's remaining budget, and therefore a carve — but one that is
        # min'd against the role-1 deadline, so it can only ever REDUCE a stage's
        # clock, never grant work that grows with ``time_limit``. Recorded rather
        # than hidden: an earlier revision passed this scan only because the
        # variable was named ``outer_deadline`` instead of a whitelisted name,
        # which is the ratchet going blind to its own author.
        "split",
    ),
    (
        "solvers/oa.py",
        "budget = remaining * _MASTER_NO_INCUMBENT_BUDGET_FRAC",
        1,
        "split",  # the OA master's share of the caller's budget
    ),
)

_KNOWN_COUNT = {(p, s): n for p, s, n, _ in KNOWN}
_KNOWN_KEYS = set(_KNOWN_COUNT)


@pytest.mark.unit
def test_no_unsaturated_role2_carve():
    """A role-2 sub-budget must saturate, or be recorded as not being one."""
    found = _scan()
    assert found, "the scanner matched nothing — it has stopped working (CLAUDE.md §6)"
    new = sorted(
        (path, line) for (path, line), n in found.items() if n > _KNOWN_COUNT.get((path, line), 0)
    )
    assert not new, (
        "unsaturated role-2 wall carve(s) — #1153.\n"
        "A sub-budget carved as a fraction of the caller's ``time_limit`` must\n"
        "stop growing at some point, or a bigger budget buys more preprocessing\n"
        "instead of more search and the answer gets WORSE (nvs19: 30 s ->\n"
        "-1098.2 over 38403 nodes, 60 s -> -1001.2 over 7619). Wrap it in\n"
        "``solver_tuning.saturate_role2(seconds, frac)``, give it a constant\n"
        "``min(...)`` ceiling, or record it in KNOWN with a category.\n"
        "(A line already in KNOWN is reported when a SECOND site lands on the same\n"
        "text — bump its recorded count only if the new site is genuinely exempt.)\n\n"
        + "\n".join(f"  {p}: {s}" for p, s in new)
    )


@pytest.mark.unit
def test_recorded_carves_still_exist():
    """The ratchet must not rot into stale bookkeeping."""
    stale = sorted(_KNOWN_KEYS - set(_scan()))
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
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            # AST, not a line grep: prose mentioning ``saturate_role2(`` inside a
            # docstring is not a call site, and this file's own doctrine is that
            # an instrument must measure what it claims to.
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            name = (
                f.id
                if isinstance(f, ast.Name)
                else (f.attr if isinstance(f, ast.Attribute) else "")
            )
            if name in _SATURATING:
                sites.append(f"{path.relative_to(_PKG)}:{node.lineno}")
    # A FLOOR, not an equality. Equality fails in both directions — including the
    # one this file's own ratchet tells an author to take, "wrap the new carve in
    # saturate_role2" — and the message would then explain only the other case.
    # Removing a wrapper is the regression; adding one is the fix.
    assert len(sites) >= 6, (
        f"the #1153 saturation count DROPPED to {len(sites)} call sites (floor 6). "
        "A carve went back to tracking the caller's budget; adding sites is fine "
        "and does not fail here.\n" + "\n".join(f"  {s}" for s in sites)
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


def _solve(path: Path, tl: float):
    """``(objective, better_is_lower)`` — the sense read from the model, never assumed."""
    model = from_nl(str(path))
    result = model.solve(time_limit=tl, gap_tolerance=1e-4)
    sense = getattr(getattr(model, "_objective", None), "sense", None)
    return result.objective, sense is None or sense is ObjectiveSense.MINIMIZE


def _worse(a: float | None, b: float | None, minimize: bool) -> bool:
    """Did quality fall going from ``a`` to ``b``, in this model's own sense?

    Hardcoding ``b > a`` reads a MAXIMIZE instance exactly backwards — real
    regressions pass and improvements fail — and nothing else in the panel would
    catch it, because every instance in the seed list happens to minimize today.
    """
    if a is None:
        return False
    if b is None:
        return True
    tol = 1e-6 * max(1.0, abs(a))
    return (b > a + tol) if minimize else (b < a - tol)


@pytest.mark.slow
@pytest.mark.correctness
@pytest.mark.skipif(not _DATA.is_dir(), reason="MINLPLib corpus not vendored")
def test_finder_entry_share_stops_the_throughput_collapse():
    """Doubling the budget must not HALVE the tree (#1153's throughput half).

    The legacy arm is run as a **control**: without it, an arm that collapsed for
    some unrelated reason — or a panel where the pump stopped being admitted at
    all — would read as a pass (CLAUDE.md §6). When the control does not
    reproduce the collapse this **skips** rather than fails, because the absence
    of the defect on a machine is not a regression in the code.

    Arms are interleaved within each repetition and the comparison is over the
    per-cell **median**, because a single OFF/ON draw is not a measurement here:
    `heatexch_gen2` is bimodal at these budgets on some machines (measured
    ``[7, 3]``, sd 2.0), and a sequential one-shot pair would redden the build on
    an unlucky draw with no defect present (CLAUDE.md §9).
    """
    reps = 2
    samples: dict[tuple[str, str, float], list[int]] = {}
    for _rep in range(reps):
        for name in _COLLAPSE_PANEL:
            path = _DATA / f"{name}.nl"
            if not path.exists():
                continue
            for tl in (_SMALL, _LARGE):
                for arm, on in (("legacy", False), ("share", True)):
                    token = _with_entry_share(on)
                    try:
                        samples.setdefault((name, arm, tl), []).append(_nodes(path, tl))
                    finally:
                        solver_tuning.reset_current(token)

    def med(name: str, arm: str, tl: float) -> float:
        return statistics.median(samples[(name, arm, tl)])

    compared = 0
    control_collapsed = 0
    failures: list[str] = []
    for name in _COLLAPSE_PANEL:
        if (name, "legacy", _SMALL) not in samples:
            continue
        compared += 1
        if med(name, "legacy", _LARGE) < med(name, "legacy", _SMALL):
            control_collapsed += 1
        if med(name, "share", _LARGE) < med(name, "share", _SMALL):
            failures.append(
                f"{name}: share ON still collapses {med(name, 'share', _SMALL)} -> "
                f"{med(name, 'share', _LARGE)} nodes ({_SMALL}s -> {_LARGE}s); "
                f"control {med(name, 'legacy', _SMALL)} -> {med(name, 'legacy', _LARGE)}; "
                f"raw {samples[(name, 'share', _SMALL)]} / {samples[(name, 'share', _LARGE)]}"
            )

    assert compared > 0, "no panel instance was vendored — this test measured nothing"
    if control_collapsed == 0:
        pytest.skip(
            "the legacy arm did not reproduce the throughput collapse on this "
            "machine, so the treatment arm is unfalsifiable here (CLAUDE.md §6); "
            "re-derive the panel with scratchpad/i1153/reps.py"
        )
    assert not failures, "\n".join(failures)


@pytest.mark.slow
@pytest.mark.correctness
@pytest.mark.skipif(not _DATA.is_dir(), reason="MINLPLib corpus not vendored")
@pytest.mark.parametrize("share_on", [False, True], ids=["legacy", "share"])
def test_incumbent_quality_is_monotone_in_time_limit(share_on):
    """More budget must never yield a worse incumbent (#1153's gate).

    Parametrised over BOTH arms on purpose. Gating only the flag-on arm would
    leave the **shipped default path** — the flag is default-off — with no
    monotonicity gate at all, which is the one configuration every user gets.
    """
    token = _with_entry_share(share_on)
    comparisons = 0
    violations: list[str] = []
    try:
        for name in _COLLAPSE_PANEL + ("tspn12",):
            path = _DATA / f"{name}.nl"
            if not path.exists():
                continue
            rung = [(tl, *_solve(path, tl)) for tl in _LADDER]
            for (tl_a, obj_a, minimize), (tl_b, obj_b, _) in zip(rung, rung[1:]):
                if obj_a is None:
                    continue  # nothing to be monotone about yet
                comparisons += 1
                if _worse(obj_a, obj_b, minimize):
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
