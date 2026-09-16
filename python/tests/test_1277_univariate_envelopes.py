"""Envelopes for the four intrinsics that used to reach the interval floor (#1277).

``entropy`` (``dm.xlogx``), ``softplus``, ``sigmoid`` and ``tan`` are first-class
discopt operators — ``canonical_expr`` emits a ``call`` node for each — but none
had an entry in ``uniform_relax._UNIVARIATE_FN``, so ``_build_univariate_call``
fell through to ``Envelope(rows=[], tight=False)``: the aux interval floor, a box
with no secant and no tangent.

The floor is SOUND, so this is a tightness bug, not a correctness one — which is
exactly why it hid. On a bare univariate objective the floor is even *exact* (the
aux is the objective, so its enclosure is the answer; measured 0.00% root gap).
The cost appears when the atom sits beside a term the composite-convex lift
cannot certify: the floor decouples ``w`` from ``t``, so the LP may take every
such term's independent minimum at once. Measured root gap, floor -> envelope
(``scripts/entry_1277_univariate_floor.py``, 18 arms, both directions in one
interleaved process):

    CALPHAD compound-energy (the #1249 model)   28.84% -> 5.45%
    coupled entropy                             73.57% -> 69.31%
    coupled softplus                             5.75% ->  2.41%
    coupled sigmoid                             14.71% ->  0.00%
    coupled tan                                 84.75% -> 22.37%

This is a BOUND-CHANGING change (CLAUDE.md §5), so the first and longest test
below is the differential soundness one: over randomized boxes, the LIFTED TRUE
POINT — ``(t, f(t))`` — must satisfy every row the builder emitted. A tighter
relaxation that cuts a feasible point is worse than no relaxation at all.
"""

from __future__ import annotations

import math

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.uniform_relax import _UNIVARIATE_FN, build_uniform_relaxation

pytestmark = [pytest.mark.smoke]

#: The four entries this issue added. Each is ``(dm builder, numpy f, domain lo)``.
OPS = {
    "entropy": (dm.xlogx, lambda t: t * np.log(t), 1e-12),
    "softplus": (dm.softplus, lambda t: np.logaddexp(0.0, t), -math.inf),
    "sigmoid": (dm.sigmoid, lambda t: 1.0 / (1.0 + np.exp(-t)), -math.inf),
    "tan": (dm.tan, np.tan, -math.inf),
}

#: The table as this module found it, so a control arm can restore one entry.
_SAVED = dict(_UNIVARIATE_FN)

#: Boxes per operator, chosen to hit the single-curvature case, the straddling
#: case, a very wide box and a nearly-degenerate one. ``tan`` additionally gets
#: boxes that cross and that touch a pole, where the verdict MUST abstain.
BOXES = {
    "entropy": [(1e-6, 1.0), (1e-3, 0.9), (0.5, 0.500001), (1e-9, 50.0), (2.0, 7.0)],
    "softplus": [(-3.0, 3.0), (-40.0, 40.0), (0.0, 1e-6), (5.0, 30.0), (-30.0, -5.0)],
    "sigmoid": [(0.0, 4.0), (-4.0, 0.0), (-6.0, 6.0), (1.0, 1.000001), (-20.0, 20.0)],
    "tan": [
        (0.05, 1.3),
        (-1.3, -0.05),
        (-0.5, 0.5),  # straddles the inflection at 0
        (0.1, 4.0),  # CROSSES the pole at pi/2 -> must abstain
        (1.0, math.pi / 2),  # TOUCHES the pole -> must abstain
        (3.3, 4.4),  # the k=1 branch, above its inflection at pi
    ],
}


def _single_atom_relaxation(builder, lo, hi):
    """``min f(x)`` over ``[lo, hi]``: columns are ``[x, w]`` with ``w = f(x)``."""
    m = dm.Model("atom")
    x = m.continuous("x", lb=lo, ub=hi)
    m.minimize(builder(x))
    rel = build_uniform_relaxation(m)
    assert rel.n_orig == 1, rel.n_orig
    assert rel.n_aux >= 1, rel.n_aux
    kinds = [k for k, _ in rel.coverage.values()]
    assert kinds == ["univariate_call"], kinds
    return rel


def _rows(rel):
    """``(A_ub, b_ub)`` as dense arrays, or ``(None, None)`` when no row was cut."""
    a = rel.model._A_ub
    if a is None:
        return None, None
    a = np.asarray(a.todense()) if hasattr(a, "todense") else np.asarray(a, dtype=float)
    return a, np.asarray(rel.model._b_ub, dtype=float)


# --------------------------------------------------------------------------- #
# Soundness first: no emitted row may cut the true point
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("op", sorted(OPS))
def test_no_emitted_row_cuts_the_lifted_true_point(op):
    """The whole risk of this change. For every box and every sample ``t`` in it,
    the lifted point ``(t, f(t))`` is feasible for the real problem, so every row
    the builder emitted must hold there — a violated row is a cut optimum."""
    builder, np_f, dom_lo = OPS[op]
    checked = 0
    for lo, hi in BOXES[op]:
        rel = _single_atom_relaxation(builder, lo, hi)
        a, b = _rows(rel)
        if a is None or a.shape[0] == 0:
            continue  # abstained: the interval floor alone, nothing to cut with
        ts = np.linspace(lo, hi, 401)
        with np.errstate(all="ignore"):
            fs = np_f(ts)
        for t, fv in zip(ts, fs):
            if not math.isfinite(float(fv)) or float(t) <= dom_lo:
                continue
            point = np.zeros(a.shape[1], dtype=float)
            point[0] = float(t)
            point[rel.n_orig] = float(fv)
            slack = a @ point - b
            worst = float(np.max(slack))
            scale = max(1.0, abs(float(fv)), abs(float(t)))
            assert worst <= 1e-7 * scale, (
                f"{op} on [{lo}, {hi}]: row cuts the true point t={t!r} f(t)={fv!r} by {worst!r}"
            )
            checked += 1
    assert checked > 500, f"{op}: only {checked} rows checked — the probe barely fired"


@pytest.mark.parametrize("op", sorted(OPS))
def test_the_aux_column_still_encloses_the_function(op):
    """The interval floor is the relaxation's last line of defence; a tightening
    that narrows the aux column past ``f``'s true range would be unsound on its
    own, independently of any row."""
    builder, np_f, dom_lo = OPS[op]
    checked = 0
    for lo, hi in BOXES[op]:
        rel = _single_atom_relaxation(builder, lo, hi)
        col_lo, col_hi = rel.model._bounds[rel.n_orig]
        ts = np.linspace(lo, hi, 2001)
        with np.errstate(all="ignore"):
            fs = np_f(ts)
        fin = fs[np.isfinite(fs) & (ts > dom_lo)]
        if not fin.size:
            continue
        scale = max(1.0, float(np.max(np.abs(fin))))
        assert col_lo <= float(np.min(fin)) + 1e-7 * scale, (op, lo, hi, col_lo)
        assert col_hi >= float(np.max(fin)) - 1e-7 * scale, (op, lo, hi, col_hi)
        checked += 1
    assert checked, f"{op}: no box produced a finite sample"


# --------------------------------------------------------------------------- #
# The curvature verdicts are the premise every row rests on
# --------------------------------------------------------------------------- #
#: An INDEPENDENT closed form of ``f''`` per operator. The table states each
#: curvature as a hand-derived sign rule (``_curv_const``, ``_curv_by_sign``,
#: ``_curv_tan``); these are what those rules are checked against.
#:
#: A sampled second difference was tried first and is the wrong instrument here:
#: on the near-degenerate boxes above (width 1e-6) the central difference of an
#: O(1) function is pure cancellation noise — it came back as +-111 for softplus,
#: which says nothing about curvature. The closed forms are exact at every width,
#: and a finite difference validates THEM on well-conditioned boxes below.
def _d2_entropy(t):
    return 1.0 / t


def _d2_softplus(t):
    s = 1.0 / (1.0 + np.exp(-t))
    return s * (1.0 - s)


def _d2_sigmoid(t):
    s = 1.0 / (1.0 + np.exp(-t))
    return s * (1.0 - s) * (1.0 - 2.0 * s)


def _d2_tan(t):
    return 2.0 * np.tan(t) / np.cos(t) ** 2


D2 = {
    "entropy": _d2_entropy,
    "softplus": _d2_softplus,
    "sigmoid": _d2_sigmoid,
    "tan": _d2_tan,
}


#: Well-conditioned boxes for validating ``D2`` itself. Deliberately NOT
#: ``BOXES``: those include boxes that touch a singularity (``entropy`` down to
#: 1e-9, ``tan`` across a pole), where the truncation error of a central
#: difference -- which scales with the FOURTH derivative -- is larger than the
#: second derivative it is meant to referee. What is checked here is the ALGEBRA
#: of ``D2``, and a benign box checks that as well as a pathological one does.
BENIGN = {
    "entropy": (0.25, 3.0),
    "softplus": (-3.0, 3.0),
    "sigmoid": (-4.0, 4.0),
    "tan": (0.05, 1.3),
}


@pytest.mark.parametrize("op", sorted(OPS))
def test_the_closed_form_second_derivative_is_itself_right(op):
    """Guard on the guard: the closed forms above are what the curvature verdicts
    are judged by, so they are judged in turn by a central difference."""
    _builder, np_f, _dom_lo = OPS[op]
    lo, hi = BENIGN[op]
    h = (hi - lo) * 1e-3
    ts = np.linspace(lo + 2 * h, hi - 2 * h, 101)
    fd = (np_f(ts + h) - 2.0 * np_f(ts) + np_f(ts - h)) / (h * h)
    cf = D2[op](ts)
    assert np.all(np.isfinite(fd)) and np.all(np.isfinite(cf)), op
    rel = np.abs(fd - cf) / np.maximum(1e-12, np.abs(cf))
    assert float(np.max(rel)) < 1e-4, (op, float(np.max(rel)))
    assert ts.size == 101


@pytest.mark.parametrize("op", sorted(OPS))
def test_a_curvature_verdict_matches_the_second_derivative_on_the_whole_box(op):
    """A "convex" verdict claims ``f'' >= 0`` over the WHOLE box, and that claim
    is what every secant and tangent row rests on. An abstention is always sound,
    so only a definite verdict is checked."""
    _builder, _np_f, dom_lo = OPS[op]
    curv_fn = _UNIVARIATE_FN[op][2]
    verdicts = 0
    for lo, hi in BOXES[op]:
        verdict = curv_fn(lo, hi)
        if verdict is None:
            continue
        ts = np.linspace(lo, hi, 20_001)
        ts = ts[ts > dom_lo]
        assert ts.size, (op, lo, hi)
        with np.errstate(all="ignore"):
            d2 = D2[op](ts)
        assert np.all(np.isfinite(d2)), (op, lo, hi, "f'' is not finite on an admitted box")
        if verdict == "convex":
            assert float(np.min(d2)) >= 0.0, (op, lo, hi, float(np.min(d2)))
        else:
            assert float(np.max(d2)) <= 0.0, (op, lo, hi, float(np.max(d2)))
        verdicts += 1
    assert verdicts >= 2, f"{op}: only {verdicts} definite verdicts to check"


def test_tan_abstains_on_every_box_that_reaches_a_pole():
    """``tan'' = 2 sec^2 t * tan t`` has the sign of ``tan`` only WITHIN a branch.
    Across a pole ``tan`` runs to +inf and returns from -inf, so the endpoint
    chord is not a chord of the graph and no secant is sound. This two-sided
    condition is why ``tan`` could not be expressed through ``domain_ok``, which
    only sees ``lo`` — it lives in ``_curv_tan``."""
    curv = _UNIVARIATE_FN["tan"][2]
    half = math.pi / 2.0
    assert curv(0.1, 4.0) is None  # crosses the pole at pi/2
    assert curv(1.0, half) is None  # touches it from the left
    assert curv(half, 2.0) is None  # touches it from the right
    assert curv(-4.0, 4.0) is None  # crosses two poles
    assert curv(-0.5, 0.5) is None  # straddles the inflection at 0
    assert curv(0.05, 1.3) == "convex"
    assert curv(-1.3, -0.05) == "concave"
    assert curv(3.3, 4.4) == "convex"  # branch k=1, above its inflection at pi
    assert curv(1.7, 3.1) == "concave"  # branch k=1, below it
    assert curv(float("nan"), 1.0) is None
    assert curv(-math.inf, 0.0) is None


def test_a_pole_crossing_tan_keeps_the_unbounded_interval_floor():
    """Abstaining must leave the aux free, not pin it to a finite box that ``tan``
    escapes."""
    rel = _single_atom_relaxation(dm.tan, 0.1, 4.0)
    assert rel.coverage[next(iter(rel.coverage))] == ("univariate_call", False)
    col_lo, col_hi = rel.model._bounds[rel.n_orig]
    assert col_lo == -math.inf and col_hi == math.inf, (col_lo, col_hi)
    a, _b = _rows(rel)
    assert a is None or a.shape[0] == 0, "no row is sound across a pole"


# --------------------------------------------------------------------------- #
# The payoff, measured against the code as it was
# --------------------------------------------------------------------------- #
def _coupled(builder, lo, hi, s, c):
    m = dm.Model("coupled")
    x = m.continuous("x", lb=lo, ub=hi)
    y = m.continuous("y", lb=lo, ub=hi)
    m.subject_to(x + y == s)
    m.minimize(builder(x) + builder(y) + c * x * y)
    return m


def _coupled_truth(np_f, lo, hi, s, c, n=400_001):
    a = np.linspace(max(lo, s - hi), min(hi, s - lo), n)
    with np.errstate(all="ignore"):
        v = np_f(a) + np_f(s - a) + c * a * (s - a)
    v = v[np.isfinite(v)]
    assert v.size, "the truth grid enclosed no finite point"
    return float(np.min(v))


COUPLED = [
    ("entropy", dm.xlogx, lambda t: t * np.log(t), 1e-6, 1.0, 1.0, 40.0),
    ("softplus", dm.softplus, lambda t: np.logaddexp(0.0, t), -3.0, 3.0, 1.0, 9.0),
    ("sigmoid", dm.sigmoid, lambda t: 1.0 / (1.0 + np.exp(-t)), 0.1, 4.0, 2.0, 5.0),
    ("tan", dm.tan, np.tan, 0.05, 1.3, 1.0, 3.0),
]


@pytest.fixture
def without_the_entries():
    """The pre-#1277 table, restored afterwards — the differential control."""
    saved = {k: _UNIVARIATE_FN.pop(k) for k in list(OPS)}
    assert len(saved) == len(OPS), sorted(saved)
    try:
        yield
    finally:
        _UNIVARIATE_FN.update(saved)
        assert all(k in _UNIVARIATE_FN for k in OPS)


@pytest.mark.parametrize("op,builder,np_f,lo,hi,s,c", COUPLED)
def test_the_envelope_tightens_the_root_bound_without_crossing_the_optimum(
    op, builder, np_f, lo, hi, s, c, without_the_entries
):
    """The §5 differential test: the new bound must be ``>=`` the old one (a
    tightening) AND ``<=`` the true optimum (still valid). Both arms run in this
    one test so the control is the same process, same box, same build."""
    truth = _coupled_truth(np_f, lo, hi, s, c)
    scale = max(1.0, abs(truth))

    old = _coupled(builder, lo, hi, s, c).solve(time_limit=60, max_nodes=1)
    old_bound = old.root_bound if old.root_bound is not None else old.bound
    assert op not in _UNIVARIATE_FN, "the control arm must not carry the entry"

    _UNIVARIATE_FN.update({k: v for k, v in _SAVED.items() if k == op})
    new = _coupled(builder, lo, hi, s, c).solve(time_limit=60, max_nodes=1)
    new_bound = new.root_bound if new.root_bound is not None else new.bound

    assert old_bound is not None and new_bound is not None
    assert new_bound >= old_bound - 1e-7 * scale, (op, old_bound, new_bound)
    assert new_bound <= truth + 1e-6 * scale, (op, new_bound, truth)
    assert new_bound > old_bound + 1e-9 * scale, (
        f"{op}: the entry changed nothing — measured a strict gain, so this is a "
        f"regression ({old_bound} -> {new_bound})"
    )


@pytest.mark.parametrize("op,builder,np_f,lo,hi,s,c", COUPLED)
def test_the_solve_still_reaches_the_true_optimum(op, builder, np_f, lo, hi, s, c):
    """Tightening the relaxation must not move the answer."""
    truth = _coupled_truth(np_f, lo, hi, s, c)
    r = _coupled(builder, lo, hi, s, c).solve(time_limit=120)
    scale = max(1.0, abs(truth))
    assert r.objective is not None
    assert r.objective >= truth - 1e-4 * scale, (op, r.objective, truth)
    assert r.objective == pytest.approx(truth, abs=1e-3 * scale), (op, r.objective, truth)
    if r.bound is not None:
        assert r.bound <= truth + 1e-6 * scale, (op, r.bound, truth)


def test_the_entries_reach_the_engine_as_tight_coverage():
    """Node counts and bounds cannot distinguish "the envelope fired" from "the
    search got lucky" (CLAUDE.md §6); the coverage verdict is the direct evidence."""
    fired = 0
    for op, (builder, _np_f, _dom) in OPS.items():
        lo, hi = BOXES[op][0]
        rel = _single_atom_relaxation(builder, lo, hi)
        kind, tight = next(iter(rel.coverage.values()))
        assert (kind, tight) == ("univariate_call", True), (op, kind, tight)
        fired += 1
    assert fired == len(OPS), fired


# --------------------------------------------------------------------------- #
# Scope of the change
# --------------------------------------------------------------------------- #
def test_the_in_repo_nl_corpus_cannot_reach_any_of_the_four():
    """The differential panel over ``python/tests/data/minlplib_nl`` is
    bound-neutral BY CONSTRUCTION, and this is the assertion that says why: `.nl`
    has no opcode for ``entropy``/``softplus``/``sigmoid``, and no instance in the
    corpus uses ``tan``. So a panel showing "no change" there is evidence of
    nothing, and the soundness evidence for this change is the row-level
    feasible-point test above, not the corpus."""
    import collections
    import pathlib

    from discopt._relax.canonical_expr import canonicalize
    from discopt.modeling.core import from_nl

    root = pathlib.Path(__file__).parent / "data" / "minlplib_nl"
    files = sorted(root.glob("*.nl"))
    assert len(files) > 50, f"expected the 60+ instance corpus, found {len(files)}"

    names: collections.Counter = collections.Counter()
    walked = 0
    for path in files:
        dag = canonicalize(from_nl(str(path)))
        roots = [r for r in [dag.objective, *dag.constraints] if r is not None]
        assert roots, f"{path.name}: canonicalized to no roots"
        seen: set[int] = set()
        stack = list(roots)
        while stack:
            node = stack.pop()
            if id(node) in seen:
                continue
            seen.add(id(node))
            if node.kind in ("call", "callN"):
                names[node.payload] += 1
            stack.extend(node.children)
        walked += len(seen)

    assert walked > 1000, f"the probe walked only {walked} nodes"
    assert set(names) == {"log", "sqrt", "exp"}, dict(names)
    assert not set(names) & set(OPS), dict(names)
