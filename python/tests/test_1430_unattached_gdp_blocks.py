"""Regression suite for #1430: a disjunctive block built on a model but never
attached to it must be REFUSED, not silently dropped.

``Model.disjunction()`` and ``Model.make_disjunct()`` are the only two
disjunctive helpers on ``Model`` that append nothing to ``_constraints``; every
other one (``either_or``, ``if_else``, ``if_then``, ``implies``, ``iff``,
``add_disjunction``) attaches. Before this fix, forgetting the attaching call
produced a **certified optimum for the relaxed problem** — the failure CLAUDE.md
§1 rules out with no slack:

    m.minimize(x + y)                                     # x, y in [0, 10]
    m.disjunction([[x >= 4, y >= 1], [y >= 6, x >= 1]])   # never attached
    m.solve()   ->  status="optimal", objective=0.0, bound=-1e-12

The true optimum of the model as written is 5.0, which is what the ``either_or``
spelling returns. The ``make_disjunct`` form is the same failure with a louder
tell: its indicator binary IS registered on the model (``num_variables`` 2 -> 4)
while its constraints are not, and the solve returned 0.0/0.0.

Structure
---------
* ``TestSilentDropIsRefused`` — the two reproducers from the issue. These FAIL
  before the fix (they return a false optimum instead of raising).
* ``TestAttachedFormsStillSolve`` — every legitimate spelling still solves, and
  to the *right* answer. A guard that made the correct forms unreachable would
  be a worse bug than the one it fixes.
* ``TestGuardDoesNotOverreach`` — the patterns the guard must NOT reject: an
  empty unattached disjunct (contributes an unused binary and nothing else), a
  factory object that never touched a solve, and nesting inside another
  unattached disjunction (one mistake, one message — not two).

Every test asserts a concrete value or a concrete message fragment; there is no
assertion here that a no-op model would satisfy (CLAUDE.md §6).
"""

from __future__ import annotations

import discopt.modeling as dm
import pytest
from discopt.modeling.core import _DisjunctiveConstraint

# The disjunction used throughout: (x >= 4 AND y >= 1) OR (y >= 6 AND x >= 1)
# over [0, 10]^2, minimizing x + y. Arm one costs 5, arm two costs 7, so the
# optimum is 5.0 -- and the relaxed model (disjunction dropped) optimizes to 0.0,
# which is the number the bug reported as "optimal".
_TRUE_OPTIMUM = 5.0
_RELAXED_OPTIMUM = 0.0


def _base():
    m = dm.Model("gdp1430")
    x = m.continuous("x", lb=0.0, ub=10.0)
    y = m.continuous("y", lb=0.0, ub=10.0)
    m.minimize(x + y)
    return m, x, y


class TestSilentDropIsRefused:
    def test_floating_disjunction_is_refused(self):
        """``m.disjunction(...)`` with no nesting: refused, naming ``either_or``.

        Before #1430 this returned ``optimal`` at 0.0 with ``bound=-1e-12`` --
        a false certificate for a problem the caller did not pose.
        """
        m, x, y = _base()
        m.disjunction([[x >= 4, y >= 1], [y >= 6, x >= 1]], name="mode")

        with pytest.raises(ValueError) as exc:
            m.solve(time_limit=15)

        msg = str(exc.value)
        assert "never added to it" in msg, msg
        assert "Model.disjunction(name='mode')" in msg, msg
        # The message must name a call that actually works, or it strands the user
        # -- ``m.subject_to(<disjunction>)`` raises TypeError, so ``either_or`` is
        # the only way in.
        assert "either_or" in msg, msg

    def test_orphan_disjunct_blocks_are_refused(self):
        """``make_disjunct`` blocks never passed to ``add_disjunction``: refused.

        Before #1430 this returned ``optimal`` at 0.0 with ``bound=0.0``. Both
        blocks are named, because a message that reported only one would have the
        user fix half the model and hit the same wrong answer again.
        """
        m, x, y = _base()
        d1 = m.make_disjunct("a")
        d1.subject_to(x >= 4)
        d1.subject_to(y >= 1)
        d2 = m.make_disjunct("b")
        d2.subject_to(y >= 6)
        d2.subject_to(x >= 1)

        with pytest.raises(ValueError) as exc:
            m.solve(time_limit=15)

        msg = str(exc.value)
        assert "Model.make_disjunct('a')" in msg, msg
        assert "Model.make_disjunct('b')" in msg, msg
        assert "add_disjunction" in msg, msg
        # The tell that distinguishes this form from the other: the indicator is
        # on the model while the rows are not.
        assert "a_active" in msg, msg

    def test_validate_refuses_without_solving(self):
        """The guard lives in ``validate``, so a caller that validates first sees
        it without paying for a solve -- and so every route through ``validate``
        (not just ``Model.solve``) is covered by one check."""
        m, x, y = _base()
        m.disjunction([[x >= 4], [y >= 6]])
        with pytest.raises(ValueError, match="never added to it"):
            m.validate()

    def test_partial_attachment_is_still_refused(self):
        """Attaching ONE of two blocks does not excuse the other.

        The dangerous shape: the model solves, the answer looks plausible, and
        exactly one arm of the intended disjunction is enforced.
        """
        m, x, y = _base()
        d1 = m.make_disjunct("kept")
        d1.subject_to(x >= 4)
        d2 = m.make_disjunct("forgotten")
        d2.subject_to(y >= 6)
        m.add_disjunction([d1])

        with pytest.raises(ValueError) as exc:
            m.solve(time_limit=15)
        msg = str(exc.value)
        assert "Model.make_disjunct('forgotten')" in msg, msg
        assert "'kept'" not in msg, msg


class TestAttachedFormsStillSolve:
    """Each legitimate spelling solves, and to the true optimum -- not the
    relaxed one. Asserting the VALUE, not merely "it did not raise": a guard that
    let the model through while the disjunction stayed dropped would pass a
    no-raise test."""

    def test_either_or(self):
        m, x, y = _base()
        m.either_or([[x >= 4, y >= 1], [y >= 6, x >= 1]])
        r = m.solve(time_limit=30)
        assert r.objective == pytest.approx(_TRUE_OPTIMUM, abs=1e-4)
        assert r.objective > _RELAXED_OPTIMUM + 1.0

    def test_add_disjunction(self):
        m, x, y = _base()
        d1 = m.make_disjunct("a")
        d1.subject_to(x >= 4)
        d1.subject_to(y >= 1)
        d2 = m.make_disjunct("b")
        d2.subject_to(y >= 6)
        d2.subject_to(x >= 1)
        m.add_disjunction([d1, d2])
        r = m.solve(time_limit=30)
        assert r.objective == pytest.approx(_TRUE_OPTIMUM, abs=1e-4)

    def test_nested_factory_disjunction(self):
        """The use ``disjunction()`` exists for: an inner disjunction nested into
        an attached one. The guard must see through the nesting, otherwise it
        breaks the only supported use of the factory."""
        m = dm.Model("nested1430")
        x = m.continuous("x", lb=0.0, ub=20.0)
        m.minimize(x)
        inner = m.disjunction([[x >= 1, x <= 3], [x >= 5, x <= 7]], name="inner")
        m.either_or([[inner], [x >= 15]], name="outer")
        r = m.solve(time_limit=30)
        assert r.objective == pytest.approx(1.0, abs=1e-4)

    def test_doubly_nested_factory_disjunction(self):
        """Two levels of nesting: the reachability walk must be transitive, not
        one level deep."""
        m = dm.Model("nested2_1430")
        x = m.continuous("x", lb=0.0, ub=20.0)
        m.minimize(-x)
        deep = m.disjunction([[x <= 3], [x >= 5, x <= 7]], name="deep")
        mid = m.disjunction([[deep], [x >= 9, x <= 11]], name="mid")
        m.either_or([[mid], [x >= 18]], name="top")
        r = m.solve(time_limit=30)
        assert r.objective == pytest.approx(-20.0, abs=1e-4)


class TestGuardDoesNotOverreach:
    def test_empty_unattached_disjunct_is_allowed(self):
        """A disjunct with no constraints contributes an unused indicator binary
        and nothing else, so it cannot change an answer. Refusing it would reject
        the legitimate pattern of building blocks in a loop and using a subset."""
        m, x, y = _base()
        m.make_disjunct("unused")  # never given a constraint, never attached
        r = m.solve(time_limit=30)
        assert r.objective == pytest.approx(_RELAXED_OPTIMUM, abs=1e-6)

    def test_factory_object_without_a_solve_is_not_refused(self):
        """``m.disjunction(...)`` on its own is still a pure factory call: it
        returns the object and raises nothing. The refusal belongs at validate
        time, where the model is about to be used."""
        m, x, y = _base()
        obj = m.disjunction([[x <= 3], [x >= 7]], name="inner")
        assert isinstance(obj, _DisjunctiveConstraint)
        assert obj not in m._constraints

    def test_nested_floater_is_reported_once_via_its_root(self):
        """An unattached disjunction nesting another unattached one is ONE
        mistake. Reporting the child separately would have the user chase a
        second message that fixing the first already resolves."""
        m, x, y = _base()
        child = m.disjunction([[x <= 3], [x >= 7]], name="child")
        m.disjunction([[child], [y >= 6]], name="parent")

        with pytest.raises(ValueError) as exc:
            m.validate()
        msg = str(exc.value)
        assert "name='parent'" in msg, msg
        assert "name='child'" not in msg, msg
        assert msg.count("Model.disjunction") == 1, msg

    def test_models_without_disjunctions_are_untouched(self):
        """The overwhelmingly common case pays nothing and changes nothing."""
        m = dm.Model("plain1430")
        x = m.continuous("x", lb=-2.0, ub=2.0)
        m.minimize((x - 0.5) ** 2)
        m.subject_to(x >= -1.0)
        r = m.solve(time_limit=15)
        assert r.objective == pytest.approx(0.0, abs=1e-6)
        assert m._unattached_gdp_blocks() == ([], [])

    def test_if_else_and_if_then_are_not_flagged(self):
        """``if_else`` builds a ``_DisjunctiveConstraint`` of its own and attaches
        it; ``if_then`` attaches an indicator constraint. Neither goes through the
        factories, so neither may be reported."""
        m = dm.Model("ifelse1430")
        x = m.continuous("x", lb=-2.0, ub=2.0)
        b = m.binary("b")
        # w = (x >= 0) ? x + 4 : x + 6 -- an if_else-built disjunction, attached
        # by if_else itself rather than by either of the two factories.
        w = m.if_else(x >= 0, x + 4.0, x + 6.0)
        m.minimize(w)
        m.if_then(b, [x >= 1.0])
        assert m._unattached_gdp_blocks() == ([], [])
        r = m.solve(time_limit=30)
        # min over x in [-2, 2]: the x < 0 arm gives x + 6 >= 4 at x = -2; the
        # x >= 0 arm gives x + 4 >= 4 at x = 0. Both reach 4.0.
        assert r.objective == pytest.approx(4.0, abs=1e-3)
