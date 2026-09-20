"""``dm.atan2``: the exact sign-definite half-plane rewrite.

Before this change discopt had no ``atan2`` at any doorway. ``dm.atan2`` did not
exist; a raw ``FunctionCall("atan2", ...)`` — reachable only through the GAMS
link and ``serialize.loads`` — got the ``[-inf, +inf]`` enclosure that
``interval_eval``'s generic multi-argument arm hands every unrecognised n-ary
call, which becomes an unbounded aux floor in ``uniform_relax._build_multivar``.
Measured on ``min atan2(y, x)`` over ``x, y in [0.5, 2]``, on ``1225337``::

    has dm.atan2 ?                 False
    raw-node enclosure:            -inf  inf
    status: feasible   certified: False   bound: None

The objective value was right; nothing could certify it.

The obstacle is real but *local*. ``atan2`` is discontinuous across the branch
cut ``{y = 0, x <= 0}``, and unlike an ordinary nonconvexity that jump does not
shrink under branching — see
:func:`test_branch_cut_range_does_not_shrink_under_refinement`, which pins the
measurement the design rests on. Away from the cut, though, ``atan2`` is
*exactly* equal to ``atan`` of a sign-definite ratio, and ``atan`` and division
are both already relaxed rigorously. So ``dm.atan2`` rewrites rather than
relaxing, and refuses loudly on the one box shape that cannot be certified
anyway.

See :mod:`discopt.modeling._atan2`.
"""

from __future__ import annotations

import math
import random

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.convexity.interval import Interval
from discopt._relax.convexity.interval_eval import evaluate_interval
from discopt.modeling._atan2 import classify_atan2, rewrite_atan2
from discopt.modeling.core import Atan2BranchCutError, FunctionCall

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _value_at(expr, point: dict) -> float:
    """Evaluate ``expr`` at a point by collapsing the box to it.

    ``evaluate_interval``'s ``box`` overrides the declared bounds, so a
    degenerate ``[v, v]`` per variable yields a point enclosure. This evaluates
    the expression *as built* — the classification already happened against the
    real declared bounds — which is exactly what the differential needs.
    """
    box = {v: Interval(np.asarray(float(p)), np.asarray(float(p))) for v, p in point.items()}
    enc = evaluate_interval(expr, None, box)
    lo = float(np.min(np.asarray(enc.lo)))
    hi = float(np.max(np.asarray(enc.hi)))
    return 0.5 * (lo + hi)


def _has_atan2_node(expr) -> bool:
    """True if any ``FunctionCall("atan2", ...)`` survives anywhere in the DAG."""
    seen: set[int] = set()
    stack = [expr]
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, FunctionCall) and node.func_name == "atan2":
            return True
        for attr in ("args", "operand", "left", "right", "base", "terms", "body"):
            child = getattr(node, attr, None)
            if child is None:
                continue
            stack.extend(child if isinstance(child, (list, tuple)) else [child])
    return False


# --------------------------------------------------------------------------- #
# the measurement the design rests on
# --------------------------------------------------------------------------- #


def test_branch_cut_range_does_not_shrink_under_refinement():
    """Refining a box ONTO the cut drives atan2's range toward 2*pi, not zero.

    This is *why* atan2 is not an atom. Spatial B&B convergence needs the
    relaxation gap to vanish as the box collapses; here the range grows. A
    future contributor tempted to add a `MathFunc::Atan2` with an envelope
    should read this test first: no envelope over a cut-straddling box can
    close a gap, because the function itself does not.
    """
    ranges = []
    for k in range(12):
        w = 2.0**-k
        vals = [
            math.atan2(-w + j * (2 * w / 200), -2.0 + i * (1.0 / 200))
            for i in range(201)
            for j in range(201)
        ]
        ranges.append(max(vals) - min(vals))

    # Monotonically non-decreasing, and converging to 2*pi from below.
    for earlier, later in zip(ranges, ranges[1:]):
        assert later >= earlier - 1e-12, (earlier, later)
    assert ranges[-1] == pytest.approx(2 * math.pi, abs=1e-5)
    assert ranges[-1] > ranges[0]


# --------------------------------------------------------------------------- #
# the differential: rewrite == atan2, pointwise
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "case,x_lb,x_ub,y_lb,y_ub",
    [
        ("x>0", 0.5, 4.0, -3.0, 3.0),  # right half-plane
        ("y>0", -4.0, 4.0, 0.25, 3.0),  # upper, spanning x<0 -- the interesting one
        ("y<0", -4.0, 4.0, -3.0, -0.25),  # lower, spanning x<0
        ("both", 0.5, 4.0, 0.5, 4.0),  # two identities hold; either is exact
        ("left-up", -5.0, -1.0, 0.1, 3.0),  # strictly LEFT of the cut, above it
        ("left-dn", -5.0, -1.0, -3.0, -0.1),  # strictly left, below
        ("x-fixed-0", 0.0, 0.0, 0.5, 2.0),  # degenerate x == 0, y > 0
    ],
)
def test_rewrite_matches_atan2_pointwise(case, x_lb, x_ub, y_lb, y_ub):
    """The emitted expression equals math.atan2 at every sampled feasible point.

    This is the test that protects soundness. A misclassified sign would not
    make the model *loose*, it would make it a different model — the rewrite is
    an equality, not a relaxation — so the only real defence is a pointwise
    differential over the declared box.
    """
    m = dm.Model()
    x = m.continuous("x", lb=x_lb, ub=x_ub)
    y = m.continuous("y", lb=y_lb, ub=y_ub)
    expr = dm.atan2(y, x)

    rng = random.Random(20260920)
    compared = 0
    worst = 0.0
    for _ in range(400):
        px = rng.uniform(x_lb, x_ub)
        py = rng.uniform(y_lb, y_ub)
        got = _value_at(expr, {x: px, y: py})
        want = math.atan2(py, px)
        worst = max(worst, abs(got - want))
        compared += 1
    # Corners too: the extremes are where a sign slip would show first.
    for px in (x_lb, x_ub):
        for py in (y_lb, y_ub):
            worst = max(worst, abs(_value_at(expr, {x: px, y: py}) - math.atan2(py, px)))
            compared += 1

    assert compared == 404, compared  # the probe fired (CLAUDE.md §6)
    assert worst < 1e-9, (case, worst)


def test_rewrite_leaves_no_atan2_node_behind():
    """The point of the rewrite: nothing downstream ever sees an atan2 node."""
    m = dm.Model()
    x = m.continuous("x", lb=0.5, ub=2.0)
    y = m.continuous("y", lb=0.5, ub=2.0)
    assert not _has_atan2_node(dm.atan2(y, x))
    # Control: the walker does find one when it is there, so the assertion above
    # is a real check and not a walker that traverses nothing.
    assert _has_atan2_node(FunctionCall("atan2", y, x))


# --------------------------------------------------------------------------- #
# the refusal
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "x_lb,x_ub,y_lb,y_ub",
    [
        (-2.0, 2.0, -2.0, 2.0),  # box contains the origin
        (-2.0, -1.0, -1.0, 1.0),  # squarely on the cut
        (-2.0, 0.0, 0.0, 1.0),  # touches the cut at y == 0 (closed, so refused)
        (0.0, 2.0, -1.0, 1.0),  # x >= 0 but not x > 0: origin still reachable
    ],
)
def test_refuses_on_cut_straddling_box(x_lb, x_ub, y_lb, y_ub):
    m = dm.Model()
    x = m.continuous("x", lb=x_lb, ub=x_ub)
    y = m.continuous("y", lb=y_lb, ub=y_ub)
    with pytest.raises(Atan2BranchCutError) as excinfo:
        dm.atan2(y, x)
    msg = str(excinfo.value)
    # The message must be actionable: name the cut, and show the bounds it read.
    assert "branch cut" in msg
    assert f"[{y_lb:g}, {y_ub:g}]" in msg
    assert f"[{x_lb:g}, {x_ub:g}]" in msg


def test_refusal_is_not_a_discontinuous_intrinsic_error():
    """atan2 is implemented; the *box* is the problem. The types must differ."""
    m = dm.Model()
    x = m.continuous("x", lb=-1.0, ub=1.0)
    y = m.continuous("y", lb=-1.0, ub=1.0)
    with pytest.raises(Atan2BranchCutError):
        dm.atan2(y, x)
    assert not issubclass(Atan2BranchCutError, dm.DiscontinuousIntrinsicError)
    assert issubclass(Atan2BranchCutError, ValueError)


def test_unbounded_default_bounds_refuse():
    """A variable with discopt's default (huge) bounds straddles zero -> refuse."""
    m = dm.Model()
    x = m.continuous("x")
    y = m.continuous("y")
    with pytest.raises(Atan2BranchCutError):
        dm.atan2(y, x)


def test_semi_infinite_bound_still_proves_a_sign():
    """An infinite endpoint is not a failure: (1, +inf) is still positive."""
    m = dm.Model()
    x = m.continuous("x", lb=1.0, ub=float("inf"))
    y = m.continuous("y", lb=-5.0, ub=5.0)
    assert classify_atan2(y, x).which == "x_pos"
    assert not _has_atan2_node(dm.atan2(y, x))


# --------------------------------------------------------------------------- #
# arrays
# --------------------------------------------------------------------------- #


def test_array_arguments_rewrite_and_keep_shape():
    m = dm.Model()
    x = m.continuous("x", shape=3, lb=0.5, ub=2.0)
    y = m.continuous("y", shape=3, lb=-1.0, ub=1.0)
    expr = dm.atan2(y, x)
    assert not _has_atan2_node(expr)
    enc = evaluate_interval(expr, None)
    assert np.asarray(enc.lo).shape == (3,)


def test_array_refused_when_any_element_straddles():
    """One bad element refuses the whole call -- the conservative direction."""
    m = dm.Model()
    x = m.continuous("x", shape=3, lb=np.array([1.0, 1.0, -1.0]), ub=np.array([2.0, 2.0, 2.0]))
    y = m.continuous("y", shape=3, lb=-1.0, ub=1.0)
    with pytest.raises(Atan2BranchCutError):
        dm.atan2(y, x)


# --------------------------------------------------------------------------- #
# the payoff: a rewritten model certifies
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
def test_rewritten_model_solves_to_a_certified_global_optimum():
    """Fails before this change: certified=False, bound=None, no dual bound.

    ``min atan2(y, x)`` over ``[0.5, 2]^2`` is attained at ``(x, y) = (2, 0.5)``,
    value ``atan(0.25)``.
    """
    m = dm.Model()
    x = m.continuous("x", lb=0.5, ub=2.0)
    y = m.continuous("y", lb=0.5, ub=2.0)
    m.minimize(dm.atan2(y, x))
    result = m.solve()

    assert result.objective == pytest.approx(math.atan(0.25), abs=1e-6)
    assert result.gap_certified is True
    assert result.bound is not None
    # The certificate invariant: the dual bound never crosses the true optimum.
    assert result.bound <= math.atan(0.25) + 1e-6


@pytest.mark.smoke
def test_rewritten_model_exports_to_nl():
    """The .nl writer refused atan2 outright; the rewrite makes the model writable.

    ``atan`` and division both have opcodes, so once atan2 is gone the same
    model round-trips through the format that previously rejected it.
    """
    from discopt.export.nl import to_nl

    m = dm.Model()
    x = m.continuous("x", lb=0.5, ub=2.0)
    y = m.continuous("y", lb=0.5, ub=2.0)
    m.minimize(dm.atan2(y, x))
    text = to_nl(m)
    assert "o" in text  # an operator section exists at all

    # Control: the raw node is still refused by the writer, so the success above
    # is the rewrite's doing and not the writer having learned atan2.
    m2 = dm.Model()
    x2 = m2.continuous("x", lb=0.5, ub=2.0)
    y2 = m2.continuous("y", lb=0.5, ub=2.0)
    m2.minimize(FunctionCall("atan2", y2, x2))
    with pytest.raises(ValueError, match="atan2"):
        to_nl(m2)


# --------------------------------------------------------------------------- #
# the residual raw node (serialize.loads / the GAMS link on a cut box)
# --------------------------------------------------------------------------- #


def test_raw_atan2_node_gets_a_finite_sound_enclosure():
    """[-pi, pi] instead of [-inf, +inf]: atan2's range, valid for any argument.

    An unbounded enclosure means an unbounded aux floor and hence no finite dual
    bound at all. A raw node can still reach the relaxation layer from
    ``serialize.loads`` or the GAMS link, so it gets the one bound that is
    unconditionally true.
    """
    m = dm.Model()
    x = m.continuous("x", lb=-2.0, ub=2.0)
    y = m.continuous("y", lb=-2.0, ub=2.0)
    enc = evaluate_interval(FunctionCall("atan2", y, x), m)
    assert float(np.min(np.asarray(enc.lo))) == pytest.approx(-math.pi)
    assert float(np.max(np.asarray(enc.hi))) == pytest.approx(math.pi)

    # Soundness spot-check: every attainable value lies inside the enclosure.
    rng = random.Random(7)
    checked = 0
    for _ in range(2000):
        v = math.atan2(rng.uniform(-2, 2), rng.uniform(-2, 2))
        assert -math.pi <= v <= math.pi
        checked += 1
    assert checked == 2000


def test_enclosure_ignores_model_argument():
    """The rewrite reads bounds with ``evaluate_interval(expr, None)``.

    ``model`` is threaded but never consulted — variable bounds come from
    ``Variable.lb``/``.ub``. If that ever changes, the sign classification would
    silently misread a bound and emit a *wrong* rewrite, so pin it here rather
    than discovering it as a false certificate.
    """
    m = dm.Model()
    x = m.continuous("x", lb=0.25, ub=4.0)
    with_model = evaluate_interval(x, m)
    without = evaluate_interval(x, None)
    assert float(with_model.lo) == float(without.lo) == 0.25
    assert float(with_model.hi) == float(without.hi) == 4.0


# --------------------------------------------------------------------------- #
# doorway consistency
# --------------------------------------------------------------------------- #


def test_gams_arctan2_rewrites_when_bounds_allow():
    from discopt.gams.instructions import _BINARY

    m = dm.Model()
    x = m.continuous("x", lb=0.5, ub=2.0)
    y = m.continuous("y", lb=-2.0, ub=2.0)
    assert not _has_atan2_node(_BINARY["arctan2"](y, x))


def test_gams_arctan2_keeps_the_node_on_a_cut_box():
    """The GAMS link must NOT start raising where it used to solve locally.

    An imported model's bounds are not the importer's to fix, and the local path
    already reports such a solve honestly as ``LocallyOptimal``. Turning that
    into an error would be a capability regression.
    """
    from discopt.gams.instructions import _BINARY

    m = dm.Model()
    x = m.continuous("x", lb=-2.0, ub=2.0)
    y = m.continuous("y", lb=-2.0, ub=2.0)
    assert _has_atan2_node(_BINARY["arctan2"](y, x))


def test_rewrite_helper_returns_none_rather_than_raising():
    """``rewrite_atan2`` is the non-raising API the GAMS doorway relies on."""
    m = dm.Model()
    x = m.continuous("x", lb=-2.0, ub=2.0)
    y = m.continuous("y", lb=-2.0, ub=2.0)
    assert rewrite_atan2(y, x) is None
    assert classify_atan2(y, x).which is None


def test_atan2_is_exported_from_both_namespaces():
    import discopt

    assert discopt.atan2 is dm.atan2
    assert discopt.Atan2BranchCutError is Atan2BranchCutError


# --------------------------------------------------------------------------- #
# the mutable-bounds guard
# --------------------------------------------------------------------------- #


def test_widening_the_assumed_bound_is_refused_at_validate():
    """Bounds are read at build time but stay mutable -- so re-check them.

    Measured before the guard existed: ``dm.atan2(y, x)`` built under
    ``y >= 0.5``, then ``y.lb = -2``, returned ``+2.356`` where ``math.atan2``
    gives ``-0.785`` -- off by pi, at 3 of 4 probe points, with no error. That
    is a false model, so ``validate()`` (and hence every ``solve()``) refuses it.
    """
    m = dm.Model()
    x = m.continuous("x", lb=0.5, ub=2.0)
    y = m.continuous("y", lb=0.5, ub=2.0)
    m.minimize(dm.atan2(y, x))
    m.validate()  # fine as built

    # Widen whichever argument the chosen identity divides by.
    denominator = m._atan2_preconditions[0][0]
    denominator.lb = -2.0

    with pytest.raises(Atan2BranchCutError, match="no longer holds"):
        m.validate()
    with pytest.raises(Atan2BranchCutError, match="no longer holds"):
        m.solve()


def test_narrowing_bounds_keeps_the_rewrite_valid():
    """Branching and FBBT only narrow, and narrowing can never break a sign."""
    m = dm.Model()
    x = m.continuous("x", lb=0.5, ub=4.0)
    y = m.continuous("y", lb=0.5, ub=4.0)
    m.minimize(dm.atan2(y, x))
    for var in (x, y):
        var.lb, var.ub = 1.0, 2.0
    m.validate()


def test_precondition_is_registered_once_per_rewrite():
    m = dm.Model()
    x = m.continuous("x", lb=0.5, ub=2.0)
    y = m.continuous("y", lb=0.5, ub=2.0)
    assert m._atan2_preconditions == []
    dm.atan2(y, x)
    dm.atan2(y, x)
    assert len(m._atan2_preconditions) == 2
    # A refused call registers nothing -- there is no rewrite to guard.
    z = m.continuous("z", lb=-1.0, ub=1.0)
    with pytest.raises(Atan2BranchCutError):
        dm.atan2(z, z)
    assert len(m._atan2_preconditions) == 2


def test_gams_doorway_rewrite_is_guarded_too():
    """Registration lives in ``build_rewrite``, so every doorway is covered."""
    from discopt.gams.instructions import _BINARY

    m = dm.Model()
    x = m.continuous("x", lb=0.5, ub=2.0)
    y = m.continuous("y", lb=-2.0, ub=2.0)
    m.minimize(_BINARY["arctan2"](y, x))
    assert len(m._atan2_preconditions) == 1
    m.validate()  # fine as built
    x.lb = -1.0
    with pytest.raises(Atan2BranchCutError, match="no longer holds"):
        m.validate()


def test_precondition_survives_a_serialize_round_trip():
    """The guard must travel with the document, or a reloaded model is unguarded.

    A round-tripped model keeps the *rewritten* expression; without the
    preconditions it would keep no record of the sign that rewrite assumed, so a
    widening applied after loading would differ from ``atan2`` by pi silently.
    """
    from discopt.serialize import dumps, loads

    m = dm.Model()
    x = m.continuous("x", lb=0.5, ub=2.0)
    y = m.continuous("y", lb=0.5, ub=2.0)
    m.minimize(dm.atan2(y, x))

    restored = loads(dumps(m))
    assert len(restored._atan2_preconditions) == 1
    restored.validate()

    # The carried denominator must alias a variable of the RESTORED model, not a
    # detached copy -- otherwise widening the restored bound would not be seen.
    denominator, _sign, _label = restored._atan2_preconditions[0]
    denominator.lb = -2.0
    with pytest.raises(Atan2BranchCutError, match="no longer holds"):
        restored.validate()


def test_document_without_preconditions_still_loads():
    """Backward compatibility: a pre-guard document has no such key."""
    import json

    from discopt.serialize import dumps, loads

    m = dm.Model()
    x = m.continuous("x", lb=0.5, ub=2.0)
    y = m.continuous("y", lb=0.5, ub=2.0)
    m.minimize(dm.atan2(y, x))

    doc = json.loads(dumps(m))
    del doc["state"]["_atan2_preconditions"]
    restored = loads(json.dumps(doc))
    assert restored._atan2_preconditions == []
    restored.validate()
