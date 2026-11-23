"""#2293: a chained integer index ``x[k][s]`` is the variable it names, everywhere.

``y = x[k]; ... y[s]`` -- the idiom of every time-stepped transcription -- builds
``IndexExpression(IndexExpression(x, k), s)``. The tape lowering resolved that to
one slot since #2283, but ``_flat_index.resolve_scalar_slot``, which every
structure layer uses (relaxation canonicalizer, term classifier, OBBT, linear
extractors), refused it. So the relaxation saw each ``y[s]`` as an *opaque*
subexpression: no envelope, and an interval-cache key holding the full box,
``2 * 8 * n`` bytes per node. On the issue's 14,336-row model the root build alone
passed 14 GB and was killed (the same model spelled ``x[k + 1, s]`` peaked at
310 MiB).

Two fixes, each pinned here:

A. The ``[]`` operator builds ``x[k][s]`` as the single node ``x[k, s]`` when the
   chain is all-integer on a ``Variable``, so every layer -- 63 sites test
   ``isinstance(e.base, Variable)`` -- sees one spelling. ``resolve_scalar_slot``
   also resolves a genuine chain (one built with ``IndexExpression(...)``
   directly), by the rule the tape uses. The check that matters is *spelling
   invariance*: a model written with ``x[k][s]`` must solve identically -- status,
   objective, bound, node count -- to the same model written ``x[k, s]``. Before
   the fix the chained spelling had opaque relaxation nodes, and on the kinetics
   model below it was declined by the native spatial kernel (the multilinear split
   did not recognize ``y[s]``), so it ran a different solver route.
B. A full-box interval-cache key shares one bytes object per distinct box instead of
   carrying its own copy, so a model that genuinely has opaque nodes cannot repeat
   the blow-up. Keys compare exactly as before, so no cache hit or miss changes.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._flat_index import resolve_scalar_slot
from discopt.modeling.core import IndexExpression, Variable

# ---------------------------------------------------------------------------
# A: the resolver
# ---------------------------------------------------------------------------


def _model():
    m = dm.Model("chain")
    a = m.continuous("a", shape=(2,), lb=0, ub=1)  # offset 0..1
    x = m.continuous("x", shape=(3, 4), lb=0, ub=1)  # offset 2..13
    z = m.continuous("z", shape=(2, 3, 2), lb=0, ub=1)  # offset 14..25
    m.minimize(a[0])
    return m, a, x, z


def _chain(base, *parts):
    """A genuine nested chain, built the way graph machinery does (not via ``[]``)."""
    for p in parts:
        base = IndexExpression(base, p)
    return base


@pytest.mark.parametrize(
    "parts, expected",
    [
        (("x", 1, 2), 2 + 1 * 4 + 2),
        (("x", -1, -1), 2 + 2 * 4 + 3),
        (("x", np.int64(2), np.int32(0)), 2 + 2 * 4 + 0),
        (("z", 1, 2, 0), 14 + 1 * 6 + 2 * 2 + 0),
        (("z", (1, 2), 1), 14 + 1 * 6 + 2 * 2 + 1),
        (("z", 1, (2, 1)), 14 + 1 * 6 + 2 * 2 + 1),
    ],
)
def test_resolver_resolves_a_genuine_chain(parts, expected):
    m, _, x, z = _model()
    var = {"x": x, "z": z}[parts[0]]
    assert resolve_scalar_slot(_chain(var, *parts[1:]), m) == expected


@pytest.mark.parametrize(
    "parts",
    [
        ("x", 1),  # partial: a row, not a slot
        ("z", 1, 2),  # partial chain
        ("x", slice(1, None), 0),  # slice in the chain
        ("x", 1, slice(1, None)),  # slice at the end
        ("x", 3, 0),  # out of range on its own axis
        ("x", 0, 4),
        ("x", -4, 0),
        ("x", True, 0),  # a bool is a mask, not an index
        ("x", 0, False),
        ("x", 1, 2, 0),  # over-indexed
    ],
)
def test_resolver_refuses_what_is_not_one_slot(parts):
    m, _, x, z = _model()
    var = {"x": x, "z": z}[parts[0]]
    assert resolve_scalar_slot(_chain(var, *parts[1:]), m) is None


@pytest.mark.parametrize(
    "build, expected",
    [
        (lambda x, z: x[1][2], 2 + 1 * 4 + 2),
        (lambda x, z: x[1, 2], 2 + 1 * 4 + 2),
        (lambda x, z: x[-1][-1], 2 + 2 * 4 + 3),
        (lambda x, z: x[np.int64(2)][np.int32(0)], 2 + 2 * 4 + 0),
        (lambda x, z: z[1][2][0], 14 + 1 * 6 + 2 * 2 + 0),
        (lambda x, z: z[1, 2][1], 14 + 1 * 6 + 2 * 2 + 1),
        (lambda x, z: z[1][2, 1], 14 + 1 * 6 + 2 * 2 + 1),
    ],
)
def test_operator_chain_is_one_node_on_the_variable(build, expected):
    m, _, x, z = _model()
    e = build(x, z)
    assert isinstance(e, IndexExpression) and isinstance(e.base, Variable)
    assert resolve_scalar_slot(e, m) == expected
    # Cross-check against numpy's own reading of the same index chain.
    flat = np.arange(26)
    xs, zs = flat[2:14].reshape(3, 4), flat[14:26].reshape(2, 3, 2)
    assert build(xs, zs) == expected


@pytest.mark.parametrize(
    "build",
    [
        lambda x, z: x[1],  # partial: a row, not a slot
        lambda x, z: z[1][2],  # partial chain
        lambda x, z: x[1:][0],  # slice in the chain
        lambda x, z: x[1][1:],  # slice at the end
        lambda x, z: x[3][0],  # out of range on its own axis
        lambda x, z: x[0][4],
        lambda x, z: x[-4][0],
        lambda x, z: x[True][0],  # a bool is a mask, not an index
        lambda x, z: x[0][False],
    ],
)
def test_operator_keeps_what_is_not_one_slot(build):
    m, _, x, z = _model()
    try:
        expr = build(x, z)
    except (IndexError, TypeError, ValueError):
        return  # refused at construction: nothing to resolve
    assert resolve_scalar_slot(expr, m) is None


def test_operator_joins_only_integer_chains_and_keeps_partial_rows():
    m, _, x, z = _model()
    row = x[1]
    assert isinstance(row.base, Variable) and row.index == 1
    assert z[1][2].index == (1, 2) and isinstance(z[1][2].base, Variable)  # partial, joined
    assert isinstance(x[1:][0].base, IndexExpression)  # slice part: not joined
    assert isinstance(x[1][1:].base, IndexExpression)
    with pytest.raises(IndexError):
        x[1][4]  # the operator's out-of-range guard still fires


def test_tape_and_structure_layers_share_one_resolver():
    """The tape fast path delegates, so the two cannot disagree on a slot."""
    from discopt._nl_expr_compiler import _static_scalar_slot

    m, _, x, z = _model()
    checked = 0
    for e in (_chain(x, 1, 2), _chain(x, -1, -1), _chain(z, 1, 2, 0), x[1:][0], x[1]):
        assert _static_scalar_slot(e, m) == resolve_scalar_slot(e, m)
        checked += 1
    assert checked == 5


def test_canonicalizer_makes_a_chain_a_variable_not_an_opaque_node():
    from discopt._relax.canonical_expr import _Canonicalizer

    m, _, x, _ = _model()
    c = _Canonicalizer(m)
    chained, direct = c.canon(_chain(x, 1, 2)), c.canon(x[1, 2])
    assert chained.kind == "var"
    assert chained is direct  # interned: the same canonical node


# ---------------------------------------------------------------------------
# A: spelling invariance of whole solves
# ---------------------------------------------------------------------------


def _ode(chained: bool):
    """Implicit-Euler kinetics in the issue's shape (shared nonconvex rate terms)."""
    N, S, h = 3, 4, 0.1
    m = dm.Model("ode")
    x = m.continuous("x", shape=(N + 1, S), lb=0, ub=2)
    p = m.continuous("p", shape=(2,), lb=-1, ub=1)
    for s in range(S):
        m.subject_to(x[0, s] == 0.5)
    for k in range(N):
        y = x[k + 1] if chained else None

        def at(s, k=k, y=y):
            return y[s] if chained else x[k + 1, s]

        rates = [p[j % 2] * at(j % S) * at((j + 1) % S) for j in range(2)]
        for s in range(S):
            m.subject_to(at(s) - (x[k][s] if chained else x[k, s]) - h * rates[s % 2] == 0)
    m.minimize(dm.sum([(x[N, s] - 0.4) ** 2 for s in range(S)]))
    return m


def _bilinear(chained: bool):
    m = dm.Model("bil")
    x = m.continuous("x", shape=(3, 3), lb=-1, ub=2)

    def at(i, j):
        return x[i][j] if chained else x[i, j]

    for i in range(3):
        m.subject_to(dm.sum([at(i, j) for j in range(3)]) == 1)
    m.minimize(dm.sum([at(i, j) * at(j, i) for i in range(3) for j in range(3) if i != j]))
    return m


def _minlp(chained: bool):
    m = dm.Model("minlp")
    y = m.binary("y", shape=(2, 3))
    x = m.continuous("x", shape=(2, 3), lb=0, ub=4)

    def at(v, i, j):
        return v[i][j] if chained else v[i, j]

    for i in range(2):
        for j in range(3):
            m.subject_to(at(x, i, j) <= 4 * at(y, i, j))
        m.subject_to(dm.sum([at(y, i, j) for j in range(3)]) <= 2)
    m.subject_to(dm.sum([at(x, i, j) for i in range(2) for j in range(3)]) >= 5)
    m.minimize(
        dm.sum([(at(x, i, j) - 1.5) ** 2 + 0.3 * at(y, i, j) for i in range(2) for j in range(3)])
        - at(x, 0, 0) * at(x, 1, 1)
    )
    return m


@pytest.mark.parametrize("build", [_ode, _bilinear, _minlp], ids=["ode", "bilinear", "minlp"])
def test_chained_and_direct_spellings_solve_identically(build):
    out = []
    for chained in (False, True):
        r = build(chained).solve(time_limit=120, max_nodes=200, deterministic=True)
        out.append(
            (str(r.status), r.objective, r.bound, r.root_bound, r.node_count, r.gap_certified)
        )
    assert out[0][0] in ("optimal", "node_limit", "feasible"), out[0]
    assert out[1] == out[0]


# ---------------------------------------------------------------------------
# B: full-box cache keys share their bytes
# ---------------------------------------------------------------------------


def _full_box_build(monkeypatch, fresh_bytes: bool = False):
    """Root relaxation with every node keyed on the full box.

    Every node is forced through the ``opaque`` arm of ``_support_box_key`` (the
    arm a genuinely opaque subtree takes), which is the arm #2293 blew up on.
    ``fresh_bytes`` restores the pre-#2293 key (a private copy per key) for the
    differential.
    """
    import discopt._relax.uniform_relax as U

    monkeypatch.setattr(U, "_node_support_cols", lambda node: None)
    if fresh_bytes:
        monkeypatch.setattr(
            U._Builder,
            "_shared_box_bytes",
            lambda self: (self._flat_lb.tobytes(), self._flat_ub.tobytes()),
        )
    m = _bilinear(chained=False)
    U.clear_analysis_cache(m)
    built = U.build_uniform_relaxation(m, (np.full(9, -1.0), np.full(9, 2.0)))
    return U, m, built


def test_full_box_keys_share_one_bytes_object_per_box(monkeypatch):
    U, m, _ = _full_box_build(monkeypatch)
    analysis = m.__dict__[U._ANALYSIS_ATTR]
    full = [k for k in analysis.bounds_by_box if len(k) == 3 and len(k[1]) == 9 * 8]
    assert len(full) >= 2, len(full)  # the probe saw several real full-box keys
    assert len({id(k[1]) for k in full}) == 1
    assert len({id(k[2]) for k in full}) == 1


def _lp_data(built):
    milp = built.model
    a = milp._A_ub
    a = a.toarray() if hasattr(a, "toarray") else np.asarray(a)
    return milp._c, a, milp._b_ub, np.asarray(milp._bounds, dtype=float)


def test_sharing_changes_no_cache_answer(monkeypatch):
    """Keys hash and compare as before, so the relaxation is byte-identical."""
    _, _, shared = _full_box_build(monkeypatch)
    _, _, fresh = _full_box_build(monkeypatch, fresh_bytes=True)
    compared = 0
    for u, v in zip(_lp_data(shared), _lp_data(fresh)):
        assert np.array_equal(u, v, equal_nan=True)
        compared += 1
    assert compared == 4
