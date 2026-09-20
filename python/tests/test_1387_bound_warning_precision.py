"""#1387 -- the large-bound warning must not round the default box into the sentinel.

``_format_bad_bound_entries`` formatted with ``%.2g``, so the default continuous
box ``-9.999e19`` printed as ``-1e+20`` -- the ``CONSTRAINT_INF`` sentinel. Those
two magnitudes sit on opposite sides of the one threshold that decides which
certificate a caller gets (#850): a bound below 1e20 is honoured as finite and
yields ``optimal`` at the corner, a bound at or beyond it is a true infinity and
yields ``unbounded``. A caller who gets ``optimal`` at ``-9.999e+19`` from an
unbounded-looking column reads this warning to find out why, and was told the
bound was the sentinel.

These tests assert the *distinguishability* -- that three materially different
bound magnitudes produce three different strings -- rather than pinning the
wording, so the message can be reworded without breaking them.
"""

from __future__ import annotations

import warnings

import pytest

from discopt import Model
from discopt.constants import CONSTRAINT_INF, DEFAULT_VARIABLE_BOUND

WARN_MATCH = "very large or infinite declared bounds"


def _bound_warning(build) -> str:
    """Solve *build()* and return the single large-bound warning it emitted."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build().solve(time_limit=10)
    messages = [str(w.message) for w in caught if WARN_MATCH in str(w.message)]
    assert messages, "the large-bound warning did not fire; the probe measured nothing"
    return messages[0]


def _default_box_model() -> Model:
    m = Model()
    x = m.continuous("x")  # no bounds -> the default box
    m.minimize(x)
    return m


def _explicit_below_default_model() -> Model:
    m = Model()
    x = m.continuous("x", lb=-9.9e19, ub=9.9e19)
    m.minimize(x)
    return m


def _explicit_sentinel_model() -> Model:
    m = Model()
    x = m.continuous("x", lb=-CONSTRAINT_INF, ub=CONSTRAINT_INF)
    m.minimize(x)
    return m


@pytest.mark.smoke
def test_the_constants_are_distinct_and_ordered():
    """The oracle for every test below, asserted rather than assumed."""
    assert DEFAULT_VARIABLE_BOUND < CONSTRAINT_INF
    # ... and close enough that two significant figures cannot separate them,
    # which is the whole defect.
    assert f"{DEFAULT_VARIABLE_BOUND:.2g}" == f"{CONSTRAINT_INF:.2g}"


@pytest.mark.smoke
def test_default_box_is_not_rendered_as_the_sentinel():
    msg = _bound_warning(_default_box_model)

    assert f"{DEFAULT_VARIABLE_BOUND:.6g}" in msg, msg
    assert "lb=-1e+20" not in msg, f"the default box is still rounded to the sentinel: {msg}"
    assert "ub=1e+20" not in msg, f"the default box is still rounded to the sentinel: {msg}"


@pytest.mark.smoke
def test_three_different_magnitudes_give_three_different_messages():
    """The point of the fix: bounds a user must tell apart must read apart."""
    default = _bound_warning(_default_box_model)
    below = _bound_warning(_explicit_below_default_model)
    sentinel = _bound_warning(_explicit_sentinel_model)

    assert default != below
    assert below != sentinel
    assert default != sentinel


@pytest.mark.smoke
def test_a_default_bound_is_labelled_as_such():
    """"You declared no bound" is the actionable fact, not the magnitude."""
    msg = _bound_warning(_default_box_model)

    assert "[default]" in msg, msg
    # and the consequence a reader is chasing is spelled out
    assert "optimal" in msg and "unbounded" in msg, msg


@pytest.mark.smoke
def test_an_explicit_bound_is_not_labelled_default():
    """A user who typed a huge bound is not told they omitted one."""
    assert "[default]" not in _bound_warning(_explicit_sentinel_model)
    assert "[default]" not in _bound_warning(_explicit_below_default_model)


@pytest.mark.smoke
def test_the_certificate_this_warning_explains_is_unchanged():
    """#1387 is a diagnostics fix; the #850 certificate it describes still stands."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = _default_box_model().solve(time_limit=10)

    assert r.status == "optimal"
    assert r.objective == pytest.approx(-DEFAULT_VARIABLE_BOUND, rel=1e-9)
