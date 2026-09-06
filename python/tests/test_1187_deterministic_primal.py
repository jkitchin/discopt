"""``deterministic=True`` must hold on the PRIMAL path too (#1187).

#1116 made ``deterministic`` a real flag and neutralized the role-2 wall budgets
it knew about. #1187 is the measurement that it did not reach all of them, and
that the survivor sat on the primal side.

``clay0303hfsg``, ``deterministic=True``, ``max_nodes=20``, ``time_limit=120``,
repetitions of the same binary in one process on unmodified ``main``::

    repeat   nodes   objective            dual bound
    1        27      55092.52061935072    -1.2309090606600805e-05
    2        27      55092.52061935072    -1.2309090606600805e-05
    3        27      55092.52061935072    -1.2309090606600805e-05
    4        27      46785.551237214655   -1.2309090604747465e-05

with a third incumbent (41573.26258778966) in a second process — a 25 % spread at
an *identical* node count, while the dual bound agreed to 12 significant figures.
So the tree was not moving; a primal heuristic was choosing differently.

The mechanism, localized by recording the caller of every ``perf_counter`` read
and diffing the sequences between a slow repetition and a fast one: the NLP-BB
root RENS budget, ``_RENS_BUDGET_FRAC * (time_limit - elapsed)`` capped at
``_RENS_BUDGET_CAP_S``, handed to a nested ``solve_model`` as its ``time_limit``.
A fraction of what is left on the clock is the textbook role-2 gate — it decides
how much of the rounding neighbourhood the sub-MINLP gets through — and it was
never routed through ``_role2_*``. As the process warmed up (the same solve went
25.5 s -> 17.0 s) the same nominal slice bought more sub-MINLP and RENS returned a
different point. Routing it through :func:`solver._role2_slice` collapsed the
25 % spread to agreement in 13 significant figures, at a *better* incumbent
(26669.11) than any wall-truncated repetition found — the #1116 result again.

Why the shape needed its own helper: ``_role2_budget`` / ``_role2_deadline``
return ``None`` and ``_role2_horizon`` returns ``math.inf``, and none of those is
a legal ``time_limit`` for a nested ``solve_model`` — removing a nested solve's
wall bound entirely would break the role-1 promise (CLAUDE.md §1). The no-clock
value there is the caller's own ``time_limit``: a constant of the model rather
than of how far into the run the machine happened to be.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import discopt.solver as solver
import discopt.solver_tuning as solver_tuning
import pytest

pytestmark = [pytest.mark.relaxation]

#: The instance and settings from the issue. Named here rather than inline so the
#: slow test below and the docstring above cannot drift apart.
_NL = Path(__file__).parent / "data" / "minlplib_nl" / "clay0303hfsg.nl"

#: Repetitions the slow test compares, after a discarded warm-up. The issue's own
#: reproduction discards the first solve: the divergence appears from repeat 2
#: onwards, because it is the warming process (JIT/allocator/cache state) that
#: changes what a fixed wall slice buys.
_REPEATS = 3

#: What "reproducible" means for this assertion.
#:
#: The 25 % primal spread the issue reports is 12 orders of magnitude outside this
#: band. A residual this fix does NOT remove is inside it: ~1.3e-14 relative,
#: alternating between two values in step with the run's wall time, with the dual
#: bound bit-identical across every repetition. Asserting bit-equality here would
#: assert something the change does not deliver; asserting a loose tolerance would
#: not catch the defect. This is the band between them, and it is deliberately far
#: tighter than the residual so a *return* of the defect cannot hide under it.
#:
#: What the residual is NOT: ``_deadline_wall_cap``'s 3 s clamp on one heuristic
#: sub-NLP, which a ``perf_counter``-caller trace pointed at (the divergence lands
#: on a ``fractional_diving`` step that clamp bounds). Suppressing it under the
#: flag left the alternation *exactly* unchanged over five repetitions — …842466
#: at 67.8/67.3 s, …842823 at 64.1/64.0/64.7 s — so that hypothesis is falsified
#: and the change was not shipped. #1187 stays open on the residual.
_OBJ_RTOL = 1e-9


def _tuning(**kw):
    return solver_tuning.set_current(dataclasses.replace(solver_tuning.SolverTuning(), **kw))


def test_role2_slice_passes_the_carved_slice_through_by_default():
    """Default OFF: the flag must not change the shipped search."""
    token = _tuning(deterministic=False)
    try:
        assert solver._role2_slice(4.0, whole=120.0) == 4.0
    finally:
        solver_tuning.reset_current(token)


def test_role2_slice_returns_the_whole_role1_budget_under_the_flag():
    """The no-clock value is the caller's own ``time_limit``, not ``None``/``inf``.

    A nested ``solve_model``'s ``time_limit`` is its own role-1 contract. Handing
    it ``None`` or ``math.inf`` would let a sub-solve run without any wall bound,
    which trades a reproducibility bug for a broken role-1 promise — the same
    trade ``SolverTuning.deterministic`` refuses for the phase-entry gates.
    """
    token = _tuning(deterministic=True)
    try:
        got = solver._role2_slice(4.0, whole=120.0)
        assert got == 120.0
        assert got not in (None, float("inf")), "a nested time_limit must stay finite"
    finally:
        solver_tuning.reset_current(token)


def test_role2_slice_is_wired_into_the_solve_path():
    """A helper with no call site is a dead flag (CLAUDE.md §3).

    Docstrings are stripped first so a mention in prose cannot stand in for a
    call — the failure mode this whole file exists to catch is a promise that is
    documented and not enforced.
    """
    body = re.sub(r'"""(?:.|\n)*?"""', "", Path(solver.__file__).read_text())
    assert body.count("_role2_slice(") - 1 >= 1, "_role2_slice has no call site"


def test_the_rens_slice_is_routed():
    """Pin the specific gate #1187 measured, so it cannot quietly lose the wrapper.

    Unwrapping it is invisible to every other test here: the helper would still
    exist, still be called elsewhere, and ``deterministic=True`` would silently
    stop covering the one site the issue is about.
    """
    src = Path(solver.__file__).read_text()
    assert "_RENS_BUDGET_FRAC" in src, "the RENS budget is gone — retarget this test"
    window = src[src.index("_rens_budget = ") : src.index("_rens_budget = ") + 600]
    assert "_role2_slice(" in window, (
        "the RENS sub-MINLP budget is no longer routed through _role2_slice — "
        "deterministic=True has stopped covering the gate #1187 measured"
    )
    assert "whole=time_limit" in window, (
        "the RENS slice's no-clock value must be the caller's own time_limit"
    )


@pytest.mark.slow
@pytest.mark.timeout(1800)
def test_clay0303hfsg_primal_is_reproducible_under_the_flag():
    """The issue's own reproduction, as a regression test.

    Fails on the pre-fix tree: three incumbents 25 % apart at 27 nodes. Passes
    after the RENS slice is routed. The first solve is discarded as a warm-up,
    exactly as the issue's reproduction says to — the divergence is driven by the
    process warming up, so comparing against a cold first run measures the warm-up
    rather than the defect.
    """
    from discopt.modeling.core import from_nl

    if not _NL.exists():  # pragma: no cover - corpus file is committed
        pytest.skip(f"corpus instance missing: {_NL}")

    rows = []
    for i in range(_REPEATS + 1):
        m = from_nl(str(_NL))
        r = m.solve(time_limit=120.0, gap_tolerance=1e-4, deterministic=True, max_nodes=20)
        if i:
            rows.append((r.node_count, r.objective, r.bound))

    # Executed-comparison count (CLAUDE.md §6): a probe that compared nothing
    # would otherwise pass as "no divergence found".
    comparisons = 0
    base_nodes, base_obj, base_bound = rows[0]
    assert base_obj is not None, "no incumbent found — the probe measured nothing"
    for nodes, obj, bound in rows[1:]:
        comparisons += 1
        assert nodes == base_nodes, f"node_count moved: {base_nodes} -> {nodes}"
        assert obj is not None, "an incumbent was lost between repetitions"
        assert abs(obj - base_obj) <= _OBJ_RTOL * abs(base_obj), (
            f"incumbent moved: {base_obj!r} -> {obj!r} "
            f"(rel {abs(obj - base_obj) / abs(base_obj):.2e} > {_OBJ_RTOL:.0e}) — "
            "a role-2 wall budget is deciding the primal answer again (#1187)"
        )
        assert bound == base_bound, f"dual bound moved: {base_bound!r} -> {bound!r}"
    assert comparisons == _REPEATS - 1, (
        f"the probe made {comparisons} comparisons, expected {_REPEATS - 1}"
    )
