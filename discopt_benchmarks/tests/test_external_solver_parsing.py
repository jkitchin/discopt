"""Parsing and refusal-handling tests for external .nl solvers.

These pin a defect found by the 2026-08 pre-release audit. ``[solvers.baron]``
carried ``command = "baron"``; on a machine with GAMS installed, PATH resolves
that to the GAMS build, which reads only ``.bar``. Handed a ``.nl`` it prints a
usage message and exits **rc = 0**, so the run recorded "BARON did not solve
this instance" rather than "BARON was never invoked correctly" -- and every
ratio computed against a solver with zero solves flatters us.

Fixing the path exposed a second bug underneath: BARON was being parsed by
Couenne's parser, which reads only Couenne's ``Lower bound:`` / ``Upper bound:``
lines. BARON prints ``Objective <value>`` instead, so it parsed to
``status=OPTIMAL, objective=None`` -- a claimed success no correctness check
could ever test against a known optimum.

The stdout fixtures below are **verbatim captures** from the real binaries, not
hand-written approximations; the sense convention in particular was established
by running both solvers on ``pointpack02`` (known optimum ``+2.0``, maximize)
and observing that BARON prints ``+2`` and Couenne prints ``-2``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import tomllib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.metrics import SolveStatus
from benchmarks.runner import BenchmarkConfig, BenchmarkRunner, SolverConfig

pytestmark = pytest.mark.unit

# Verbatim: /Applications/AMPL/baron pointpack02.nl  (maximize, true optimum +2.0)
BARON_MAXIMIZE_STDOUT = (
    "BARON 25.12.10 (2025.12.10): 0 iterations, numerical difficulties.\n"
    "Objective 2.0000000007597647\n"
)

# Verbatim: /Applications/AMPL/baron ex3_1_2.nl  (minimize, true optimum -30665.5387)
BARON_OPTIMAL_STDOUT = (
    "BARON 25.12.10 (2025.12.10): 1 iterations, optimal within tolerances.\n"
    "Objective -30665.53867365177\n"
)

# Verbatim: GAMS.framework/Versions/53/Resources/baron ex3_1_2.nl -- the wrong
# binary, which exits rc=0.
GAMS_BARON_USAGE_STDOUT = (
    " Usage : /Library/Frameworks/GAMS.framework/Versions/53/Resources/baron"
    " [-f] input(.bar)\n"
    " Usage Contd. : The -f option prints the files\n"
)

# Verbatim: /Applications/AMPL/baron alkylation.nl -- 10 vars, 11 constraints.
BARON_DEMO_REFUSAL_STDOUT = (
    "\nSorry, a demo license is limited to 10 variables and\n"
    "10 constraints and objectives.\n"
    "You have 10 variables, 11 constraints, and 1 objective.\n\n"
    "Contact us at <licensing@ampl.com> or https://discuss.ampl.com/\n"
)

# Verbatim: /Applications/AMPL/couenne pointpack02.nl -- same maximize instance,
# reported in INTERNAL MINIMIZATION sense.
COUENNE_MAXIMIZE_STDOUT = (
    "couenne: Optimal\n"
    "Lower bound:                                   -2\n"
    "Upper bound:                                   -2  (gap: 0.00%)\n"
    "Branch-and-bound nodes:                         0\n"
)


def _runner() -> BenchmarkRunner:
    return BenchmarkRunner(BenchmarkConfig(suite_name="unit"))


def _baron_cfg() -> SolverConfig:
    return SolverConfig(name="baron", command="/Applications/AMPL/baron", solver_type="external")


def test_baron_objective_is_parsed_at_all():
    """The original bug: OPTIMAL with objective=None, so nothing could be checked."""
    res = _runner()._parse_baron("ex3_1_2", "baron", BARON_OPTIMAL_STDOUT, 0.02)
    assert res.status is SolveStatus.OPTIMAL
    assert res.objective == pytest.approx(-30665.53867365177)


def test_baron_objective_is_original_sense_on_a_maximize_model():
    """BARON reports original sense; negating it would invert the answer.

    ``bchoco06`` is vendored and maximizes, so ``_nl_is_maximize`` resolves
    without the MINLPLib cache -- the negation path Couenne needs is live here,
    and must still leave BARON's value alone.
    """
    r = _runner()
    assert r._nl_is_maximize(r._find_nl_file("bchoco06")), (
        "bchoco06 must parse as a maximize model or this test proves nothing"
    )
    res = r._parse_baron("bchoco06", "baron", BARON_MAXIMIZE_STDOUT, 0.02)
    assert res.objective == pytest.approx(2.0000000007597647), (
        "BARON's Objective line is original-sense and must not be negated"
    )


def test_couenne_bounds_are_still_negated_for_maximize():
    """Guard the fix above did not disturb Couenne's (correct) convention."""
    r = _runner()
    res = r._parse_ampl_solver("bchoco06", "couenne", COUENNE_MAXIMIZE_STDOUT, 0.02)
    assert res.objective == pytest.approx(2.0), (
        "Couenne reports internal-minimization sense; -2 must come back as +2"
    )


@pytest.mark.parametrize(
    "stdout,label",
    [
        (GAMS_BARON_USAGE_STDOUT, "wrong binary printing usage at rc=0"),
        (BARON_DEMO_REFUSAL_STDOUT, "license refusing the instance size"),
    ],
)
def test_a_solver_that_never_read_the_model_is_an_error(stdout, label):
    r = _runner()
    parsed = r._parse_external_output("baron", "ex3_1_2", stdout, "", 0.01)
    res = r._reject_non_result(_baron_cfg(), "ex3_1_2", parsed, stdout)
    assert res.status is SolveStatus.ERROR, f"{label} must be ERROR, got {res.status}"


def test_claimed_success_without_an_objective_is_an_error():
    """The class guard, exercised through a solver that is not BARON.

    Any external adapter that reports OPTIMAL with nothing to check is rejected
    -- that is the shape a correctness gate must never silently accept.
    """
    r = _runner()
    parsed = r._parse_ampl_solver("ex3_1_2", "couenne", "couenne: Optimal\n", 0.01)
    assert parsed.status is SolveStatus.OPTIMAL and parsed.objective is None
    res = r._reject_non_result(
        SolverConfig(name="couenne", command="couenne", solver_type="external"),
        "ex3_1_2",
        parsed,
        "couenne: Optimal\n",
    )
    assert res.status is SolveStatus.ERROR


def test_baron_command_is_an_absolute_path_not_a_bare_name():
    """PATH resolution is the root cause; a bare name reintroduces it.

    Machine-independent on purpose: it asserts the *shape* of the setting, not
    the particular path this developer's AMPL bundle happens to live at.
    """
    cfg_path = Path(__file__).resolve().parents[1] / "config" / "benchmarks.toml"
    cfg = tomllib.load(cfg_path.open("rb"))
    command = cfg["solvers"]["baron"]["command"]
    assert command.startswith("/"), (
        f"[solvers.baron].command is {command!r}; a bare name resolves through "
        "PATH, which on a GAMS machine finds the .bar-only build"
    )
