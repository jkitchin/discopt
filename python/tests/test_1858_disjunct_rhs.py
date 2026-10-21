"""#1858: a raw ``Constraint(body, sense, rhs)`` with ``rhs != 0`` in a disjunct or
indicator row.

``Constraint`` is stored normalized (``body sense 0``) and top-level rows with a
non-zero ``rhs`` are refused by ``subject_to`` and ``Model.validate`` (#909). Rows
inside ``either_or`` / ``disjunction`` / ``if_then`` / ``Disjunct`` blocks were not
checked, and the GDP lowerings read only ``body``: ``Constraint(x, "<=", 2.0)`` in a
disjunct was solved as ``x <= 0`` and certified (big-M and mbigm for ``either_or``,
every method for ``if_then``). Such rows are now refused at the call that adds
them, and again by ``validate`` for rows that reach the model another way.

The oracle for each model is its hand-computed optimum; the normalized twin of
every refused model is solved alongside as a control.
"""

import random
import warnings

import discopt.modeling as dm
import pytest
from discopt.modeling.core import Constraint, _DisjunctiveConstraint, _IndicatorConstraint

pytestmark = pytest.mark.smoke

RHS_ERR = "non-zero rhs"
METHODS = ["big-m", "hull", "mbigm", "binary_multiplication"]


def _solve(m, method):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return m.solve(gdp_method=method, time_limit=60)


def _either_or_model(raw):
    # max x, x in [0, 3], (x <= 2) or (x <= -1): optimum 2.
    m = dm.Model("eo")
    x = m.continuous("x", lb=0, ub=3)
    if raw:
        rows = [[Constraint(body=x, sense="<=", rhs=2.0)], [Constraint(x, "<=", -1.0)]]
    else:
        rows = [[x <= 2.0], [x <= -1.0]]
    m.either_or(rows, name="d")
    m.maximize(x)
    return m


def _if_then_model(raw, active_value=1):
    # max x, x in [0, 3], indicator forced active, then x <= 2: optimum 2.
    m = dm.Model("it")
    x = m.continuous("x", lb=0, ub=3)
    z = m.binary("z")
    row = Constraint(body=x, sense="<=", rhs=2.0) if raw else (x <= 2.0)
    if active_value == 1:
        m.if_then(z, [row])
        m.subject_to(z >= 1)
    else:
        m.if_then(1 - z, [row])
        m.subject_to(z <= 0)
    m.maximize(x)
    return m


# -- the two repros in the issue: refused, and the normalized twin is right ----


def test_either_or_raw_rhs_refused():
    with pytest.raises(ValueError, match=RHS_ERR):
        _either_or_model(raw=True)


def test_if_then_raw_rhs_refused():
    with pytest.raises(ValueError, match=RHS_ERR):
        _if_then_model(raw=True)


def test_if_then_active_zero_raw_rhs_refused():
    with pytest.raises(ValueError, match=RHS_ERR):
        _if_then_model(raw=True, active_value=0)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    "build",
    [
        lambda: _either_or_model(raw=False),
        lambda: _if_then_model(raw=False),
        lambda: _if_then_model(raw=False, active_value=0),
    ],
    ids=["either_or", "if_then", "if_then_active0"],
)
def test_normalized_twin_solves_to_oracle(build, method):
    r = _solve(build(), method)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(2.0, abs=1e-6)


# -- every sense, and every way a row enters a disjunct ------------------------


@pytest.mark.parametrize("sense", ["<=", ">=", "=="])
def test_every_sense_refused_in_either_or(sense):
    m = dm.Model()
    x = m.continuous("x", lb=0, ub=3)
    with pytest.raises(ValueError, match=RHS_ERR):
        m.either_or([[Constraint(x, sense, 1.5)], [x <= 0]])
    assert m._constraints == []


@pytest.mark.parametrize("sense", ["<=", ">=", "=="])
def test_every_sense_refused_in_if_then(sense):
    m = dm.Model()
    x = m.continuous("x", lb=0, ub=3)
    z = m.binary("z")
    with pytest.raises(ValueError, match=RHS_ERR):
        m.if_then(z, [x <= 3, Constraint(x, sense, 1.5)])
    # Atomic: the valid row before the bad one was not recorded either.
    assert m._constraints == []


def test_nested_disjunction_raw_rhs_refused():
    m = dm.Model()
    x = m.continuous("x", lb=0, ub=3)
    with pytest.raises(ValueError, match=RHS_ERR):
        m.disjunction([[Constraint(x, "<=", 1.0)], [x >= 2]], name="inner")
    assert m._gdp_factory_disjunctions == []


def test_nested_disjunction_mutated_after_construction_refused_at_solve():
    # The construction-time check passes, then a raw row is put into the inner
    # disjunct list; ``validate`` walks the nesting and still refuses it.
    m = dm.Model()
    x = m.continuous("x", lb=0, ub=3)
    inner = m.disjunction([[x <= 1], [x >= 2]], name="inner")
    m.either_or([[inner], [x <= 0.5]], name="outer")
    inner.disjuncts[0].append(Constraint(x, ">=", 0.75))
    m.maximize(x)
    with pytest.raises(ValueError, match=RHS_ERR):
        m.validate()


@pytest.mark.parametrize("sense", ["<=", ">=", "=="])
def test_disjunct_block_raw_rhs_refused(sense):
    m = dm.Model()
    x = m.continuous("x", lb=0, ub=3)
    a = m.make_disjunct("A")
    with pytest.raises(ValueError, match=RHS_ERR):
        a.subject_to([x <= 3, Constraint(x, sense, 1.0)])
    assert a._constraints == []


def test_nested_disjunct_block_raw_rhs_refused():
    m = dm.Model()
    x = m.continuous("x", lb=0, ub=3)
    a = m.make_disjunct("A")
    inner = m.make_disjunct("A1")
    with pytest.raises(ValueError, match=RHS_ERR):
        inner.subject_to(Constraint(x, "<=", 1.0))
    a.add_disjunction([inner, m.make_disjunct("A2")], name="nest")


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("entry", ["indicator", "disjunctive"])
def test_direct_append_refused_by_validate(entry, method):
    # A row that never passed a construction-time check (``_constraints.append``)
    # is still refused before any lowering reads its ``body``.
    m = dm.Model()
    x = m.continuous("x", lb=0, ub=3)
    raw = Constraint(x, "<=", 2.0)
    if entry == "indicator":
        z = m.binary("z")
        m._constraints.append(_IndicatorConstraint(indicator=z, constraint=raw))
        m.subject_to(z >= 1)
    else:
        m._constraints.append(_DisjunctiveConstraint(disjuncts=[[raw], [x <= -1]]))
    m.maximize(x)
    with pytest.raises(ValueError, match=RHS_ERR):
        _solve(m, method)


def test_zero_rhs_raw_rows_still_accepted():
    # The refusal is about a non-zero rhs, not about building rows by hand.
    m = dm.Model()
    x = m.continuous("x", lb=0, ub=3)
    z = m.binary("z")
    m.either_or([[Constraint(x - 2, "<=", 0.0)], [Constraint(x + 1, "<=", 0.0)]])
    m.if_then(z, [Constraint(x - 2.5, "<=", 0.0)])
    m.subject_to(z >= 1)
    m.maximize(x)
    r = _solve(m, "big-m")
    assert r.status == "optimal"
    assert r.objective == pytest.approx(2.0, abs=1e-6)


# -- if_else: a raw condition's rhs, and a raw ``>=`` condition's complement ---


def test_if_else_raw_rhs_condition_refused():
    m = dm.Model()
    x = m.continuous("x", lb=0, ub=3)
    with pytest.raises(ValueError, match=RHS_ERR):
        m.if_else(Constraint(x, ">=", 1.0), 5.0, 0.0)


@pytest.mark.parametrize("x_val, expected", [(2.0, 5.0), (0.0, 0.0)])
def test_if_else_raw_ge_condition_complement(x_val, expected):
    # ``Constraint(x - 1, ">=", 0.0)`` keeps its ``>=`` sense (the operators would
    # normalize to ``<=``). Its complement was built as ``-(x - 1) <= 0`` -- the
    # condition itself -- so both branches were selectable everywhere and the
    # minimum of the two values won: 0 at x = 2, where the answer is 5.
    m = dm.Model()
    x = m.continuous("x", lb=x_val, ub=x_val)
    w = m.if_else(Constraint(x - 1, ">=", 0.0), 5.0, 0.0)
    m.minimize(w)
    r = _solve(m, "hull")
    assert r.status == "optimal"
    assert r.objective == pytest.approx(expected, abs=1e-6)


# -- the #1857 review's random battery, seed 41 --------------------------------


def _row(rng, xs, kind, raw):
    n = len(xs)
    i, j = rng.sample(range(n), 2)
    k = [t for t in range(n) if t not in (i, j)]
    xk = xs[k[0]] if k else xs[i]

    def c():
        return round(rng.uniform(-2, 2), 2)

    r = round(rng.uniform(0.5, 6), 2)
    if kind == "lin":
        return sum(c() * x for x in xs) + c() <= r * 0.5
    if kind == "lin_ge":
        return sum(c() * x for x in xs) >= -r * 0.3
    if kind == "quad":
        return sum((x - c()) ** 2 for x in xs) <= r
    if kind == "quad_ge":
        return sum((x - c()) ** 2 for x in xs) >= r
    if kind == "bilin":
        return xs[i] * xs[j] + c() * xk**2 <= r * 0.5
    if kind == "fold":
        return (xs[i] ** 2 + 3) - (2 - xs[j]) + 1.5 - (xk - (0.5 - xk**2)) <= r
    if kind == "neg":
        return -(xs[i] - xs[j] ** 2) + (-xk) >= -r
    if kind == "sq_cross":
        return (xs[i] + xs[j]) ** 2 + xk <= r
    if kind == "expcross":
        return dm.exp(0.3 * xs[i] * xs[j]) + xk - xs[i] <= r
    if kind == "rawge":
        body, rhs = xs[i] ** 2 + c() * xs[j] - xk, -r * 0.5
        return Constraint(body, ">=", rhs) if raw else body >= rhs
    if kind == "rawle":
        body, rhs = xs[i] ** 2 + xs[j] ** 2 + c() * xk + 2.0, r + 2.0
        return Constraint(body, "<=", rhs) if raw else body <= rhs
    raise KeyError(kind)


_KINDS = ["lin", "lin_ge", "quad", "quad_ge", "bilin", "fold", "neg", "sq_cross"]
_KINDS += ["expcross", "rawge", "rawle"]


def _battery_model(seed, raw):
    rng = random.Random(seed)
    m = dm.Model(f"rnd{seed}")
    n = rng.choice([2, 3, 3])
    xs = []
    for t in range(n):
        lb = rng.choice([-2, -1, 0])
        ub = rng.choice([2, 3])
        if rng.random() < 0.25:
            xs.append(m.integer(f"x{t}", lb=lb, ub=ub))
        else:
            xs.append(m.continuous(f"x{t}", lb=lb, ub=ub))

    def disjunct():
        return [_row(rng, xs, rng.choice(_KINDS), raw) for _ in range(rng.choice([1, 1, 2]))]

    shared = None
    dcs = []
    for d in range(rng.choice([1, 2])):
        ds = [disjunct() for _ in range(rng.choice([2, 2, 3]))]
        if rng.random() < 0.25:
            ds[0].append(m.disjunction([disjunct(), disjunct()], name=f"inner{d}"))
        if rng.random() < 0.2:
            shared = ds[0][0]
            ds[1].append(shared)
        dcs.append(m.either_or(ds, name=f"d{d}"))
    if rng.random() < 0.3:
        z = m.binary("z")
        m.if_then(z, disjunct())
        m.subject_to(z + sum(0.1 * x for x in xs) >= 0.2)
    if rng.random() < 0.25:
        a, b = m.make_disjunct("A"), m.make_disjunct("B")
        a.subject_to(disjunct())
        b.subject_to(disjunct())
        m.add_disjunction([a, b], name="blk")
    if len(dcs) == 2 and rng.random() < 0.5:
        y0, y1 = dcs[0].indicators, dcs[1].indicators
        choice = rng.choice(["implies", "atleast", "exactly"])
        if choice == "implies":
            m.logical(y0[0].implies(y1[1]))
        elif choice == "atleast":
            m.logical(dm.atleast(1, y0[1], y1[0]))
        else:
            m.logical(dm.exactly(1, y0[0], y1[0]))
    if shared is not None and rng.random() < 0.5 and float(shared.rhs) == 0.0:
        m.subject_to(shared)
    obj = sum(round(rng.uniform(-2, 2), 2) * x for x in xs)
    if rng.random() < 0.3:
        obj = obj + 0.3 * xs[0] ** 2
    m.minimize(obj)
    return m


def test_seed41_raw_model_refused():
    # On main this model certified 2.0089 under big-m, hull and mbigm.
    with pytest.raises(ValueError, match=RHS_ERR):
        _battery_model(41, raw=True)


@pytest.mark.parametrize("method", ["big-m", "hull", "mbigm"])
def test_seed41_normalized_twin(method):
    # Same rows written with the operators: all three lowerings agree on 0.45839,
    # the optimum the issue reports for this instance.
    r = _solve(_battery_model(41, raw=False), method)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(0.458385, abs=1e-4)
