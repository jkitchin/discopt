"""Issue #1362: a time limit handed to Ipopt must not be silently dropped.

``max_wall_time`` is an Ipopt >= 3.14 option. Against an older library (Ubuntu
24.04 still ships 3.11.9) ``cyipopt.Problem.add_option`` raises ``TypeError`` --
but only *after* the C library has printed

    Tried to set Option: max_wall_time. It is not a valid option.

on stdout. ``nlp_ipopt.solve_nlp`` caught that ``TypeError`` and logged it at
DEBUG, so the caller's time limit stopped being applied and the only trace of it
was a line of noise in every notebook and script that touched the Ipopt path.

3.11/3.12/3.13 spell the same cap ``max_cpu_time``, so the limit is translated
before the call (the message comes from C; catching the exception afterwards
cannot unprint it), and an option this Ipopt genuinely does not know is now
reported at WARNING rather than swallowed.
"""

from __future__ import annotations

import logging

import pytest
from discopt.solvers import nlp_ipopt


class _FakeCyipopt:
    def __init__(self, version):
        if version is not None:
            self.IPOPT_VERSION = version


@pytest.fixture
def fake_version(monkeypatch):
    """Pin the version ``_ipopt_version`` reads, with or without cyipopt installed."""

    def _set(version):
        monkeypatch.setitem(__import__("sys").modules, "cyipopt", _FakeCyipopt(version))

    return _set


@pytest.mark.parametrize("version", [(3, 11, 9), (3, 12, 13), (3, 13, 4)])
def test_pre_314_ipopt_gets_the_limit_as_max_cpu_time(fake_version, version):
    """The limit survives translation instead of being dropped."""
    fake_version(version)
    out = nlp_ipopt._translate_options({"max_wall_time": 7.5, "print_level": 0})
    assert "max_wall_time" not in out, "the rejected spelling must not reach Ipopt"
    assert out["max_cpu_time"] == 7.5
    assert out["print_level"] == 0


@pytest.mark.parametrize("version", [(3, 14, 0), (3, 14, 16), (3, 15, 0)])
def test_314_and_newer_keep_max_wall_time(fake_version, version):
    """A library that knows the option gets the wall-clock cap, not a CPU cap."""
    fake_version(version)
    out = nlp_ipopt._translate_options({"max_wall_time": 7.5})
    assert out == {"max_wall_time": 7.5}


def test_explicit_max_cpu_time_wins(fake_version):
    """A caller that named ``max_cpu_time`` itself is not overridden."""
    fake_version((3, 11, 9))
    out = nlp_ipopt._translate_options({"max_wall_time": 7.5, "max_cpu_time": 2.0})
    assert out["max_cpu_time"] == 2.0
    assert "max_wall_time" not in out


def test_unknown_version_is_not_assumed_old(fake_version):
    """No version, no translation -- guessing either way would be a silent change."""
    fake_version(None)
    assert nlp_ipopt._ipopt_version() == ()
    out = nlp_ipopt._translate_options({"max_wall_time": 7.5})
    assert out == {"max_wall_time": 7.5}


def test_options_without_a_time_limit_pass_through(fake_version):
    fake_version((3, 11, 9))
    opts = {"print_level": 0, "tol": 1e-8}
    assert nlp_ipopt._translate_options(opts) == opts


def test_rejected_option_is_reported_at_warning(caplog, monkeypatch):
    """A dropped option is visible. DEBUG made it invisible; that was the bug."""
    cyipopt = pytest.importorskip("cyipopt")
    import numpy as np
    from discopt.solvers import SolveStatus

    class _Problem:
        def __init__(self, *args, **kwargs):
            pass

        def add_option(self, key, value):
            if key == "definitely_not_an_ipopt_option":
                raise TypeError("Error while assigning an option")

        def solve(self, x0):
            return np.zeros_like(x0), {"status": 0, "obj_val": 0.0}

    monkeypatch.setattr(cyipopt, "Problem", _Problem)

    class _Evaluator:
        n_variables = 1
        n_constraints = 0
        variable_bounds = (np.array([-1.0]), np.array([1.0]))

        def objective(self, x):
            return 0.0

        def gradient(self, x):
            return np.zeros(1)

        def constraints(self, x):
            return np.empty(0)

        def jacobian(self, x):
            return np.empty(0)

    with caplog.at_level(logging.WARNING, logger=nlp_ipopt.__name__):
        result = nlp_ipopt.solve_nlp(
            _Evaluator(),
            np.zeros(1),
            options={"definitely_not_an_ipopt_option": 1.0},
        )

    assert result.status == SolveStatus.OPTIMAL
    warnings = [
        r.getMessage()
        for r in caplog.records
        if r.levelno >= logging.WARNING and "definitely_not_an_ipopt_option" in r.getMessage()
    ]
    assert warnings, "a dropped option must not be silent"
    assert "NOT applied" in warnings[0]
