"""Sweep D: model walks must be linear in the DAG, not in its paths or its depth.

A model that reuses a subexpression -- explicit-Euler single shooting,
``e = e + h*(u - e**2)`` -- has ``2**K`` root-to-leaf paths through ``K`` steps
but only ``O(K)`` nodes. Several walks expanded every path, so a 22-step model
overran ``time_limit=10`` by 43 s and a 40-step one never finished:

* Rust ``convert_expr`` (``model_to_repr``) had no memo; interning made the
  result right but the walk still visited every path. It also recursed on the
  right operand, so a right-nested chain ``x1 + (x2 + (...))`` deep enough
  overflowed the thread stack and killed the interpreter.
* ``core._iter_model_leaves`` (the cross-model ownership guard) had no visited set.
* ``factorable_reform._scan_for_translated_monomial`` had no visited set, and
  walked every factor of every product before counting whether three were
  non-constant, which made a deep recursive chain cubic.
* ``term_classifier._distribute_unbudgeted`` / ``_distribute_within_budget``
  re-distributed a shared node once per path.

Fast after; each case below did not finish (or crashed) before.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import discopt.modeling as dm
import pytest
from discopt import Model, _rust
from discopt._relax import factorable_reform as fr
from discopt._relax import term_classifier as tc
from discopt.modeling.core import _iter_model_leaves

K = 60  # 2**60 paths: anything that walks paths cannot finish


def _shoot(k: int = K):
    m = Model("shoot")
    u = m.continuous("u", shape=(k,), lb=-1, ub=1)
    e = u[0] * 0 + 0.5
    for i in range(k):
        e = e + 0.1 * (u[i] - e**2)
    m.subject_to(e >= 0.2)
    m.minimize(dm.sum(u * u) + e)
    return m, u, e


@pytest.mark.smoke
@pytest.mark.timeout(60)
def test_model_to_repr_shared_dag():
    m, _, _ = _shoot()
    assert _rust.model_to_repr(m) is not None


@pytest.mark.smoke
@pytest.mark.timeout(60)
def test_iter_model_leaves_visits_each_node_once():
    m, u, e = _shoot()
    leaves = list(_iter_model_leaves(e))
    assert leaves == [u]  # one Variable object, yielded once


@pytest.mark.smoke
@pytest.mark.timeout(60)
def test_translated_monomial_scan_shared_dag():
    _, _, e = _shoot()
    assert fr._scan_for_translated_monomial(e) is False


@pytest.mark.smoke
@pytest.mark.timeout(60)
def test_translated_monomial_scan_still_finds_one():
    m = Model("tm")
    x = m.continuous("x", shape=(4,), lb=0, ub=5)
    prod = (x[0] - 3) * (x[1] - 1) * (x[2] - 2)
    # Shared: the same product reached twice must still be found.
    assert fr._scan_for_translated_monomial(prod + prod) is True


@pytest.mark.smoke
@pytest.mark.timeout(60)
def test_distribute_products_shared_dag():
    _, _, e = _shoot(30)  # over the term budget: exercises the budgeted walk
    assert tc.estimate_distributed_terms(e) > tc._DISTRIBUTE_TERM_BUDGET
    out = tc.distribute_products(e)
    assert out is not None


@pytest.mark.smoke
def test_distribute_unbudgeted_reuses_the_shared_result():
    m = Model("share")
    x = m.continuous("x", shape=(2,), lb=-1, ub=1)
    s = (x[0] + x[1]) * x[0]
    out = tc._distribute_unbudgeted(s + s)
    # Both occurrences come back as one object, not two equal copies.
    assert out.left is out.right


@pytest.mark.smoke
@pytest.mark.timeout(120)
@pytest.mark.parametrize("nest", ["right", "left"])
def test_model_to_repr_deep_chain_does_not_overflow(nest):
    # In a subprocess: before the fix the right-nested chain overflowed the Rust
    # thread stack and the interpreter died with SIGSEGV.
    src = textwrap.dedent(
        f"""
        import sys
        sys.setrecursionlimit(1_000_000)
        from discopt import Model, _rust
        n = 50_000
        m = Model("deep")
        x = m.continuous("x", shape=(n,), lb=-1, ub=1)
        e = x[0]
        for i in range(1, n):
            e = {"x[i] + e" if nest == "right" else "e + x[i]"}
        m.subject_to(e <= 1.0)
        m.minimize(x[0])
        assert _rust.model_to_repr(m) is not None
        print("ok")
        """
    )
    r = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True, timeout=110)
    assert r.returncode == 0 and r.stdout.strip() == "ok", (r.returncode, r.stderr[-2000:])
