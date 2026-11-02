"""#2101: distinct but identical atoms kept the #2090 cancellation.

``_exact_polynomial`` keyed an opaque atom (``exp``, ``log``, ...) by object
identity, so ``M*dm.exp(z) + dm.exp(z) - M*dm.exp(z)`` -- three separate ``exp``
objects -- looked like three monomials and was left as written; pointwise float
evaluation then lost the middle term and certified wrong optima (5.0 and 0.405
against ln 3). Atoms are now keyed structurally (``_structural_id``).

The other direction is the soundness risk: two atoms that are *not* the same
expression must never share a key, or the rewrite would drop a term.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.modeling._exact_polynomial import _structural_id, canonicalize_expression

LN3 = float(np.log(3.0))


@pytest.mark.smoke
@pytest.mark.parametrize("B", [2.0**53, 2.0**54])
@pytest.mark.parametrize("s", [+1, -1])
def test_distinct_exp_atoms_certify_the_true_optimum(B, s):
    """The table from #2101: exactly exp(z) <= 3, so max z = ln 3."""
    m = dm.Model()
    z = m.continuous("z", lb=0, ub=5)
    body = B * dm.exp(z) + s * dm.exp(z) - B * dm.exp(z) + (1 - s) * dm.exp(z)
    m.subject_to(body <= 3)
    m.maximize(z)
    r = m.solve(time_limit=60, threads=1)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(LN3, rel=1e-6)
    assert r.bound >= LN3 - 1e-6


@pytest.mark.unit
def test_distinct_identical_atoms_are_rewritten_like_a_shared_one():
    m = dm.Model()
    z = m.continuous("z", lb=0, ub=5)
    B = 2.0**54
    shared = dm.exp(z)
    one = canonicalize_expression(B * shared + shared - B * shared)
    many = canonicalize_expression(B * dm.exp(z) + dm.exp(z) - B * dm.exp(z))
    # Both collapse to the single atom ``exp(z)`` (coefficient exactly 1).
    assert str(one) == str(many) == "exp(z)"


@pytest.mark.unit
def test_structural_ids_match_only_identical_trees():
    m = dm.Model()
    z = m.continuous("z", lb=0, ub=5)
    w = m.continuous("w", lb=0, ub=5)
    x = m.continuous("x", shape=(3,), lb=0, ub=1)
    p = m.parameter("p", value=2.0)
    memo: dict = {}

    def sid(e):
        return _structural_id(e, memo)

    assert sid(dm.exp(z)) == sid(dm.exp(z))
    assert sid(dm.exp(p * x[1])) == sid(dm.exp(p * x[1]))
    # Different trees never share an id.
    distinct = [
        dm.exp(z),
        dm.exp(w),
        dm.exp(2 * z),
        dm.log(z),
        dm.exp(z + 1),
        dm.exp(1 + z),  # commuted operands: not merged (conservative)
        dm.exp(x[0]),
        dm.exp(x[1]),
        dm.exp(-z),
        abs(z),
        z**0.5,
        z**1.5,
    ]
    ids = [sid(e) for e in distinct]
    assert len(set(ids)) == len(ids)


@pytest.mark.unit
def test_signed_zero_constants_are_not_merged():
    """``0.0`` and ``-0.0`` differ (``1/-0.0`` is ``-inf``): compared by bytes."""
    from discopt.modeling.core import BinaryOp, Constant

    m = dm.Model()
    z = m.continuous("z", lb=0.1, ub=5)
    memo: dict = {}
    a = BinaryOp("/", z, Constant(0.0))
    b = BinaryOp("/", z, Constant(-0.0))
    assert _structural_id(a, memo) != _structural_id(b, memo)


@pytest.mark.unit
def test_tagged_and_custom_nodes_keep_identity_keys():
    """A ``register_function`` lowering carries a tag; it must not merge with an
    untagged twin, and two calls of an arbitrary callable are never merged."""
    from discopt.operators import register_function

    m = dm.Model()
    z = m.continuous("z", lb=0.1, ub=1)
    f = register_function("sq_plus_one_2101", lambda u: u * u + 1, replace=True)
    memo: dict = {}
    tagged = f(z)
    assert hasattr(tagged, "_discopt_atom")
    assert _structural_id(tagged, memo) != _structural_id(z * z + 1, memo)
    assert _structural_id(tagged, memo) != _structural_id(f(z), memo)


@pytest.mark.unit
def test_a_different_atom_is_never_absorbed():
    """``M*exp(z) + exp(2z) - M*exp(z)``: exp(z)'s pair cancels exactly and exp(2z)
    is its own monomial, so nothing may be rewritten (a false merge would drop it)."""
    m = dm.Model()
    z = m.continuous("z", lb=0, ub=5)
    B = 2.0**54
    body = B * dm.exp(z) + dm.exp(2 * z) - B * dm.exp(z)
    assert canonicalize_expression(body) is body


@pytest.mark.unit
def test_structural_id_is_iterative_on_a_deep_atom_argument():
    m = dm.Model()
    x = m.continuous("x", shape=(5000,), lb=0, ub=1)
    chain = x[0]
    for i in range(1, 5000):
        chain = chain + x[i]
    memo: dict = {}
    assert _structural_id(dm.exp(chain), memo) == _structural_id(dm.exp(chain), memo)
