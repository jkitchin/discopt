"""#1565: presolve walks over a shared expression DAG must be linear in its size.

A reduced-space NN embedding (OMLT ``ReducedSpaceSmoothNNFormulation``) is one
expression in which every hidden activation is referenced by every unit of the
next layer, so its *tree* is exponentially larger than its *DAG*: a width-20,
depth-3 sigmoid net is 2,889 distinct nodes but 148,206 tree nodes. The
bound/error walks in ``gdp_reformulate`` (reached from factorable denominator
clearing), the clearable-denominator search and the structure-cut recognizer's
SymPy translation all walked it as the tree, which is how ``solve(time_limit=30)``
ran 66 s on a width-25 net and more than 10 minutes on a width-50 one.

These tests pin the walks to the DAG -- by counting calls, not by timing -- and
check the memoised results are exactly those of a walk over the same expression
re-built *without* sharing (which is what the old tree walk computed).
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax import factorable_reform as fr
from discopt._relax import gdp_reformulate as g
from discopt.modeling.core import BinaryOp, FunctionCall, UnaryOp


def _children(node):
    if isinstance(node, BinaryOp):
        return (node.left, node.right)
    if isinstance(node, UnaryOp):
        return (node.operand,)
    if isinstance(node, FunctionCall):
        return tuple(node.args)
    return ()


def _dag_and_tree_size(root) -> tuple[int, int]:
    seen: dict[int, object] = {}
    stack = [root]
    while stack:
        n = stack.pop()
        if id(n) in seen:
            continue
        seen[id(n)] = n
        stack.extend(_children(n))
    tree: dict[int, int] = {}

    def size(n):
        k = id(n)
        if k not in tree:
            tree[k] = 1 + sum(size(c) for c in _children(n))
        return tree[k]

    return len(seen), size(root)


def _unshare(node):
    """Rebuild ``node`` as a tree: a fresh interior node per occurrence."""
    if isinstance(node, BinaryOp):
        return BinaryOp(node.op, _unshare(node.left), _unshare(node.right))
    if isinstance(node, UnaryOp):
        return UnaryOp(node.op, _unshare(node.operand))
    if isinstance(node, FunctionCall):
        return FunctionCall(node.func_name, *[_unshare(a) for a in node.args])
    return node  # Variable / Constant leaves are shared in both forms


def _reduced_space_net(
    width: int, depth: int = 3, seed: int = 0, denom: bool = False, act: str = "sigmoid"
):
    """A reduced-space sigmoid network ``y == net(x)`` as one shared expression.

    With ``denom`` the output also divides by a positive affine expression of the
    last hidden layer, so the clearable-denominator search has real work. With
    ``act="gauss"`` the activation ``exp(-z**2)`` has no division at all, so that
    search finds nothing and must visit the whole expression to say so.
    """
    rng = np.random.default_rng(seed)
    m = dm.Model("net")
    x = [m.continuous(f"x{i}", lb=-1.0, ub=1.0) for i in range(2)]
    y = m.continuous("y", lb=-50.0, ub=50.0)
    h = list(x)
    for _ in range(depth):
        W = rng.normal(size=(width, len(h)))
        b = rng.normal(size=width)
        nxt = []
        for j in range(width):
            z = float(b[j])
            for i, hi in enumerate(h):
                z = z + float(W[j, i]) * hi
            nxt.append(1.0 / (1.0 + dm.exp(-z)) if act == "sigmoid" else dm.exp(-(z**2)))
        h = nxt
    out = 0.0
    for j, hj in enumerate(h):
        out = out + float(rng.normal()) * hj
    if denom:
        d = 1.0
        for hj in h:
            d = d + 0.5 * hj  # sigmoid outputs in (0, 1): d in [1, 1 + width/2]
        out = out / d
    m.subject_to(y == out)
    m.minimize(y)
    return m, m._constraints[0].body


def _count_calls(monkeypatch, module, name):
    real = getattr(module, name)
    calls = [0]

    def counting(*args, **kwargs):
        calls[0] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(module, name, counting)
    return calls


def test_net_is_a_dag_much_smaller_than_its_tree():
    _, body = _reduced_space_net(12)
    dag, tree = _dag_and_tree_size(body)
    assert tree > 10 * dag, (dag, tree)  # the premise every test below relies on


@pytest.mark.parametrize("name", ["_bound_error", "_bound_expression"])
def test_bound_walks_visit_each_shared_node_once(monkeypatch, name):
    m, body = _reduced_space_net(12)
    dag, tree = _dag_and_tree_size(body)
    calls = _count_calls(monkeypatch, g, name)
    if name == "_bound_error":
        g.bound_expression_error(body, m)
    else:
        g._bound_expression(body, m)
    # Each node is entered once per *parent edge* (a memo hit returns at once),
    # so the count is bounded by a small multiple of the DAG -- never the tree.
    assert 0 < calls[0] <= 4 * dag, (calls[0], dag, tree)


def test_clearable_denominator_search_visits_each_shared_node_once(monkeypatch):
    """The search descends the additive structure; share it and it must not
    re-search a node it already found nothing in. ``s_{k+1} = s_k + s_k`` is
    2**k additive leaves in 2k nodes, and none of its divisions is clearable
    (the denominator changes sign), so the search is exhaustive."""
    m = dm.Model("doubling")
    x = m.continuous("x", lb=1.0, ub=2.0)
    y = m.continuous("y", lb=-1.0, ub=1.0)
    s = x / (y + 0.5)  # denominator spans [-0.5, 1.5]: not sign-definite
    for _ in range(16):
        s = s + s
    dag, tree = _dag_and_tree_size(s)
    assert tree > 1000 * dag, (dag, tree)
    calls = _count_calls(monkeypatch, fr, "_find_clearable_denominator")
    assert fr._find_clearable_denominator(s, m) is None
    assert 0 < calls[0] <= 4 * dag, (calls[0], dag, tree)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("denom", [False, True])
@pytest.mark.parametrize("act", ["sigmoid", "gauss"])
def test_memoised_walks_equal_the_unshared_tree_walk(seed, denom, act):
    """Bound-neutral: the shared walk returns exactly what the tree walk did."""
    m, body = _reduced_space_net(4, seed=seed, denom=denom, act=act)
    interior: dict[int, object] = {}
    stack = [body]
    while stack:
        n = stack.pop()
        if id(n) in interior or not isinstance(n, (BinaryOp, UnaryOp, FunctionCall)):
            continue
        interior[id(n)] = n
        stack.extend(_children(n))
    checks = 0
    for b in interior.values():  # every shared subexpression, root included
        tb = _unshare(b)
        assert g._bound_expression(b, m) == g._bound_expression(tb, m)
        assert g.bound_expression_error(b, m) == g.bound_expression_error(tb, m)
        assert str(fr._find_clearable_denominator(b, m)) == str(
            fr._find_clearable_denominator(tb, m)
        )
        checks += 3
    assert checks >= 3 * 50, checks


def test_denominator_clearing_still_fires_on_shared_net():
    m, _ = _reduced_space_net(6, denom=True)
    out = fr.factorable_reformulate(m, clear_only=True)
    assert out is not m  # the positive denominator was found and cleared


# --- structure-cut recognizer (SymPy) --------------------------------------------

sympy = pytest.importorskip("sympy")


def test_to_sympy_translates_each_shared_node_once(monkeypatch):
    from discopt._relax.symbolic import cut_recognizer as cr

    cr._ensure_sympy()
    _, body = _reduced_space_net(12)
    dag, tree = _dag_and_tree_size(body)
    calls = _count_calls(monkeypatch, cr, "_to_sympy")
    cr._to_sympy(body, {})
    assert 0 < calls[0] <= 4 * dag, (calls[0], dag, tree)


def test_to_sympy_equals_unshared_translation():
    from discopt._relax.symbolic import cut_recognizer as cr

    cr._ensure_sympy()
    for seed in (0, 1):
        _, body = _reduced_space_net(4, seed=seed, denom=True)
        assert cr._to_sympy(body, {}) == cr._to_sympy(_unshare(body), {})


def test_recognizer_skips_model_translation_without_objective_products(monkeypatch):
    """The net has a nonlinear equality (so the cheap pre-check passes) but its
    objective ``y`` has no ``x*(y**k - 1)`` term: no cut is derivable, and the
    whole-model translation must not run to find that out."""
    from discopt._relax.symbolic import cut_recognizer as cr

    m, _ = _reduced_space_net(12)
    assert cr.has_square_difference_candidate(m)

    def must_not_run(*a, **k):
        raise AssertionError("model_to_sympy ran for an objective with no product term")

    monkeypatch.setattr(cr, "model_to_sympy", must_not_run)
    assert cr.recognize_and_derive_cuts(m) == []


def _gas_model():
    from discopt.benchmarks.problems.gas_network_minlp import build_gas_network_minlp

    return build_gas_network_minlp()


def test_recognizer_deadline_aborts_before_injecting_anything():
    from discopt._relax.symbolic import cut_recognizer as cr

    m = _gas_model()
    n_cons = len(m._constraints)
    with pytest.raises(cr.RecognizerDeadline):
        cr.recognize_and_inject(m, deadline=lambda: True)
    assert len(m._constraints) == n_cons  # untouched
    # and without a deadline the same model still yields its cuts
    m2 = _gas_model()
    w = next(v for v in m2._variables if v.name == "w_cs")
    w.ub = np.full(w.shape, 74.0)  # the box the cut is provable on (see test_cut_recognizer)
    assert cr.recognize_and_inject(m2, deadline=lambda: False) == 2


# --- the factorable lift (one huge constraint) -------------------------------------


def _model_snapshot(m) -> tuple:
    return (
        [v.name for v in m._variables],
        [id(c) for c in m._constraints],
        id(m._objective),
    )


def test_lift_deadline_fires_inside_one_constraint_and_leaves_model_untouched(monkeypatch):
    """The net is ONE constraint, so the #1456 per-constraint check passes before
    the lift starts and is never reached again. The clock is made to expire as
    the lift starts walking the constraint: the reform must notice inside the
    walk, drop everything built, and hand back the very model it was given."""
    m, _ = _reduced_space_net(8)
    before = _model_snapshot(m)
    armed = [False]
    consulted_after_arming = [0]

    def deadline():
        if armed[0]:
            consulted_after_arming[0] += 1
        return armed[0]

    real = fr._lift_expr

    def expiring(expr, model, lifter):
        armed[0] = True  # the clock runs out mid-lift (after the per-constraint check)
        return real(expr, model, lifter)

    monkeypatch.setattr(fr, "_lift_expr", expiring)
    out = fr.factorable_reformulate(m, deadline=deadline)
    assert armed[0], "the lift never started -- the probe fired nothing"
    assert consulted_after_arming[0] >= 1, "deadline never consulted inside the lift"
    assert out is m
    assert _model_snapshot(m) == before


def test_lift_with_unexpired_deadline_equals_lift_without_one():
    """Consulting the clock must not change what the lift builds."""
    m1, _ = _reduced_space_net(6, seed=1)
    m2, _ = _reduced_space_net(6, seed=1)
    a = fr.factorable_reformulate(m1)
    b = fr.factorable_reformulate(m2, deadline=lambda: False)
    assert a is not m1 and b is not m2
    assert [(v.name, float(v.lb), float(v.ub)) for v in a._variables] == [
        (v.name, float(v.lb), float(v.ub)) for v in b._variables
    ]
    assert [str(c.body) for c in a._constraints] == [str(c.body) for c in b._constraints]


def test_lift_convexity_gate_runs_once_per_shared_call(monkeypatch):
    m, body = _reduced_space_net(10)
    distinct_calls: dict[int, object] = {}
    stack = [body]
    while stack:
        n = stack.pop()
        if id(n) in distinct_calls:
            continue
        distinct_calls[id(n)] = n
        stack.extend(_children(n))
    n_calls = sum(1 for n in distinct_calls.values() if isinstance(n, FunctionCall))
    calls = _count_calls(monkeypatch, fr, "_should_lift_call_arg")
    out = fr.factorable_reformulate(m)
    assert out is not m
    # Aux defining equalities add a few call nodes of their own; the tree has
    # orders of magnitude more call occurrences than the DAG.
    assert 0 < calls[0] <= 2 * n_calls, (calls[0], n_calls)


def test_call_power_scan_visits_each_shared_node_once(monkeypatch):
    m, body = _reduced_space_net(12)
    dag, tree = _dag_and_tree_size(body)
    calls = _count_calls(monkeypatch, fr, "_scan_for_liftable_call_power")
    assert fr._scan_for_liftable_call_power(body, m) is False
    assert 0 < calls[0] <= 4 * dag, (calls[0], dag, tree)


def test_expr_node_count_is_the_tree_size_computed_on_the_dag():
    _, body = _reduced_space_net(12)
    dag, tree = _dag_and_tree_size(body)
    assert tree > 10 * dag
    assert fr._expr_node_count(body) == tree


def test_collect_variables_on_shared_dag_equals_unshared():
    for seed in (0, 1):
        m, body = _reduced_space_net(5, seed=seed, denom=True)
        a = g._collect_variables(body)
        b = g._collect_variables(_unshare(body))
        assert list(a) == list(b) and [id(v) for v in a.values()] == [id(v) for v in b.values()]


def test_in_loop_incumbent_obbt_deadline_never_exceeds_time_limit(monkeypatch):
    """The in-B&B incumbent-cutoff OBBT (Phase C) is clamped to the solve's clock.

    It used to receive ``now + 5.0`` whatever was left of ``time_limit``, so an
    incumbent found just before the limit let it run up to 5 s past it (measured on
    an OMLT reduced-space net: 35.4 s at ``time_limit=30``). With ``time_limit=3`` the
    old deadline is always more than 5 s after the solve began; the clamped one is at
    most ``time_limit`` after it.
    """
    import time
    from pathlib import Path

    from discopt._relax import obbt as ob

    orig = ob.obbt_tighten_root
    seen: list[float] = []

    def _spy(*args, **kwargs):
        if "incumbent_cutoff" in kwargs and kwargs.get("rounds") == 2:
            assert kwargs.get("deadline") is not None
            seen.append(float(kwargs["deadline"]))
        return orig(*args, **kwargs)

    monkeypatch.setattr(ob, "obbt_tighten_root", _spy)
    nl = Path(__file__).parent / "data" / "minlplib_nl" / "ex1222.nl"
    m = dm.from_nl(str(nl))
    T = 3.0
    t0 = time.perf_counter()
    m.solve(time_limit=T)
    assert seen, "the in-loop incumbent-cutoff OBBT never ran; the test checked nothing"
    for d in seen:
        assert d - t0 <= T + 0.5, f"Phase C deadline {d - t0:.2f}s after start, limit {T}s"
