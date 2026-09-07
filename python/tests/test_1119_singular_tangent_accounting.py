"""#1119 — the singular-tangent binding accounting, and why it did not become a gate.

#1115 shipped ``SolverTuning.singular_tangent`` default-OFF: sound, bound-helpful
on ``kriging_peaks``, but +25.6 % wall on ``eq6_1`` and +54.2 % on ``maxmin`` for
no bound at all. #1119 proposed gating the separator on whether the recovered
facet BINDS rather than on the operator that produced it, and required the
observable to be measured before the gate was written.

It was, and it falsified the gate — see ``docs/dev/performance-plan.md`` §17. The
two instances that pay have the two HIGHEST binding rates (1.0000 and 0.9580) and
the two that gain have the two lowest (0.9173 and 0.8509), and ``eq6_1`` has zero
non-binding rows, so a rule that drops non-binding rows drops none of the 22 874
it exists to remove. The direction is dead; the flag stays default-OFF.

These tests pin the instrument that produced that verdict, so the number can be
re-derived rather than re-argued. They assert self-consistency and read-only-ness,
never a specific row count: the counts move with the relaxation and are evidence,
not contract.
"""

from __future__ import annotations

import dataclasses

import discopt._relax.mccormick_lp as mlp
import numpy as np
import pytest
from discopt.modeling.core import from_nl
from discopt.solver_tuning import SolverTuning

# In-repo instances that emit singular-tangent rows through the LAZY separator
# (the eager anchor is a different code path). These are probes, not targets:
# nothing here asserts anything specific to any of them, and any in-repo instance
# that emits rows would serve — which is why this is a LIST and the tests take the
# first member that actually emits.
#
# It used to be the single name ``tspn12``, and that coupled the instrument to
# something irrelevant to it: WHICH optimal vertex the node LP happens to land on.
# The rows come from the lazy separator's trigger at the LP solution, so on a
# degenerate face any change to the pivot path — a pricing rule, a refactorization
# cadence, a scaling tweak, #1013's cost perturbation — can move the solve to an
# equally optimal vertex where the trigger is not met and the probe falls silent
# on that one instance while the separator is perfectly healthy on others.
# (Measured: with #1013's perturbation ON, ``tspn12`` reports 0 rows while
# ``tspn08`` still reports 1, and the two arms agree on status, node count and
# incumbent, with the bound differing in the 8th significant digit.) A run that
# silences the separator on EVERY candidate still fails, so the §6 guard is
# widened rather than weakened.
_EMITTING_CANDIDATES = ("tspn12", "tspn08", "tspn10", "tspn05")


def _solve_with_stats(stem: str, *, on: bool, max_nodes: int = 20, time_limit: float = 30.0):
    """Solve *stem* and return (result, summed stats over every relaxer built).

    One relaxer serves the whole tree in the common case, but the solver builds
    several for its own purposes, so the stats are summed rather than read off a
    single instance -- reading one would silently under-report.
    """
    built = []
    original = mlp.MccormickLPRelaxer.__init__

    def spy(self, *args, **kwargs):
        original(self, *args, **kwargs)
        built.append(self)

    mlp.MccormickLPRelaxer.__init__ = spy
    try:
        tuning = dataclasses.replace(
            SolverTuning(),
            singular_tangent=on,
            singular_tangent_lazy=on,
            deterministic=True,
        )
        path = _nl_path(stem)
        res = from_nl(str(path)).solve(time_limit=time_limit, max_nodes=max_nodes, tuning=tuning)
    finally:
        mlp.MccormickLPRelaxer.__init__ = original

    total = {"nodes": 0, "calls": 0, "specs": 0, "rounds": 0, "rows": 0, "binding": 0}
    for relaxer in built:
        stats = relaxer.singular_tangent_stats()
        for key in total:
            total[key] += stats[key]
    total["relaxers"] = len(built)
    return res, total


def _nl_path(stem: str):
    import pathlib

    path = pathlib.Path(__file__).parent / "data" / "minlplib_nl" / f"{stem}.nl"
    if not path.exists():
        pytest.skip(f"{stem}.nl not in the in-repo corpus")
    return path


def _first_emitting():
    """The first candidate whose flag-ON solve emits rows, with its stats.

    Cached so the flag-OFF control uses the same instance without re-solving.
    Raises rather than skipping if no candidate emits: "nothing emitted anywhere"
    is the finding this instrument exists to report, not a reason to pass.
    """
    if _first_emitting.cache is None:
        seen = {}
        for stem in _EMITTING_CANDIDATES:
            res, stats = _solve_with_stats(stem, on=True)
            seen[stem] = stats
            if stats["rows"] >= 1:
                _first_emitting.cache = (stem, res, stats)
                break
        else:
            raise AssertionError(
                "no in-repo candidate emitted a singular-tangent row "
                f"({seen}); the separator emits nothing anywhere, so every "
                "binding fraction below it is a fraction over zero rows "
                "(CLAUDE.md §6)"
            )
    return _first_emitting.cache


_first_emitting.cache = None


@pytest.mark.correctness
def test_the_separator_records_what_it_emits():
    """With the flag ON the accounting must be non-empty and internally consistent.

    An accounting that reports zero rows on every instance that could emit them is
    the CLAUDE.md §6 failure this whole instrument exists to avoid, so the row
    count is asserted non-zero before any fraction derived from it is believed.
    """
    stem, res, stats = _first_emitting()
    assert res.status in ("optimal", "feasible", "node_limit", "time_limit"), res.status

    assert stats["calls"] >= 1, (
        f"the separator never ran on {stem} ({stats}) — with the flag ON and a "
        "registered spec this test is measuring nothing"
    )
    assert stats["rows"] >= 1, (
        f"the separator ran {stats['calls']}x but emitted no rows ({stats}); a binding "
        "fraction over zero rows is not a measurement"
    )
    assert 0 <= stats["binding"] <= stats["rows"], stats
    assert stats["rounds"] >= 1 and stats["nodes"] >= 1, stats


@pytest.mark.correctness
def test_the_flag_off_arm_records_nothing():
    """The control must be uncontaminated: no rows, and therefore no fraction.

    #1119's panel would read a shared-state leak as "the feature costs nothing",
    which is the most flattering possible wrong answer.
    """
    stem, _, _ = _first_emitting()
    _, stats = _solve_with_stats(stem, on=False)
    assert stats["rows"] == 0 and stats["binding"] == 0, (
        f"flag-OFF emitted singular-tangent rows ({stats}) — the arms share state"
    )
    assert stats["calls"] == 0, f"flag-OFF ran the separator ({stats})"


@pytest.mark.unit
def test_the_tally_is_read_only():
    """It must not touch the rows it measures, and must not raise on a bad shape.

    The tally is called from inside the separator's ``try/except Exception``, so a
    raise there would silently abandon separation and move a bound. A shape
    mismatch therefore returns without recording -- but the panel above asserts a
    non-zero row count, so "recorded nothing" cannot pass as "nothing bound".
    """

    class _Probe:
        pass

    probe = _Probe()
    for attr, value in (
        ("_st_rows", 0),
        ("_st_binding", 0),
        ("_st_recent", __import__("collections").deque(maxlen=8)),
    ):
        setattr(probe, attr, value)

    rows = [[1.0, 0.0], [0.0, 1.0]]
    rhs = [1.0, 2.0]
    rows_before = [list(r) for r in rows]
    rhs_before = list(rhs)
    x = np.array([1.0, 0.5])

    record = mlp.MccormickLPRelaxer._record_singular_tangent_hits
    assert record(probe, rows, rhs, x) is None
    assert rows == rows_before and rhs == rhs_before, "the tally mutated its inputs"
    assert probe._st_rows == 2, probe._st_rows
    # x1 = 1.0 hits `x1 <= 1`; x2 = 0.5 leaves `x2 <= 2` slack by 1.5.
    assert probe._st_binding == 1, probe._st_binding

    checked = 0
    for bad_rows, bad_rhs, bad_x in (
        ([[1.0, 0.0]], [1.0], np.array([[1.0, 0.5]])),  # x not 1-D
        ([[1.0, 0.0]], [1.0, 2.0], x),  # rhs/rows disagree
        ([[1.0, 0.0, 0.0]], [1.0], x),  # width disagrees
        ([[1.0, 0.0]], [1.0], None),  # no solution
    ):
        before = probe._st_rows
        assert record(probe, bad_rows, bad_rhs, bad_x) is None
        assert probe._st_rows == before, "a malformed batch was recorded"
        checked += 1
    assert checked == 4, "the malformed-input probe stopped running (rule 6)"
