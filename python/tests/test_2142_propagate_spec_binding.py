"""#2142 item 3: the kernel's per-node FBBT fixpoint, exposed for offline comparison.

``_rust.propagate_spec_py`` runs ``propagate_spec_fixpoint`` on one box, and the
``node_observer`` record now carries the node's pre-propagation box (``pre_lo`` /
``pre_hi``), its post-propagation box (``lo`` / ``hi``) and the cutoff propagation ran
under. Together they let an instrument replay the tree's propagation on captured boxes
and compare it against a reference FBBT or another solver (the cstr big-M comparison in
``docs/dev/performance-plan.md`` §105). Both are instrumentation: nothing reaches a
solve path, which the replay test pins by requiring the binding to reproduce the tree's
own boxes bit for bit.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt import _rust
from discopt._relax.spatial_producer import build_spatial_kernel_spec

pytestmark = pytest.mark.smoke


def _spec():
    m = dm.Model("prop2142")
    x = m.continuous("x", lb=0.0, ub=4.0)
    y = m.continuous("y", lb=0.0, ub=4.0)
    z = m.binary("z")
    m.minimize(-x * y + 0.5 * x**2 - y + 2.0 * z)
    m.subject_to(x + y <= 5.0)
    m.subject_to(x * y >= 1.0)
    m.subject_to(y <= 1.0 + 3.0 * z)
    spec = build_spatial_kernel_spec(m)
    assert spec is not None, "the kernel must take this model"
    return {k: v for k, v in spec.items() if not k.startswith("meta_")}


def test_propagate_spec_py_tightens_through_cutoff_row_and_product():
    """The tanksize chain in miniature (the Rust unit test of the same name, through the
    binding): cutoff x0 <= 2 => w <= 2 via w <= x0, then x1, x2 <= 2 by reverse division."""
    n = 4
    spec = dict(
        n_cols=n,
        n_orig=n,
        c=np.array([1.0, 0.0, 0.0, 0.0]),
        integrality=np.zeros(n, dtype=np.int64),
        global_lo=np.zeros(n),
        global_hi=np.full(n, 100.0),
        fixed_row_ptr=np.array([0, 2], dtype=np.int64),
        fixed_cols=np.array([3, 0], dtype=np.int64),
        fixed_coeffs=np.array([1.0, -1.0]),
        fixed_rhs=np.array([0.0]),
        term_kind=np.zeros(0, dtype=np.int64),
        term_i=np.zeros(0, dtype=np.int64),
        term_j=np.zeros(0, dtype=np.int64),
        term_out=np.zeros(0, dtype=np.int64),
        term_p=np.zeros(0, dtype=np.int64),
        term_coeff=np.zeros(0),
        term_cst=np.zeros(0),
        blf_w=np.array([3], dtype=np.int64),
        blf_a_ptr=np.array([0, 1], dtype=np.int64),
        blf_a_cols=np.array([1], dtype=np.int64),
        blf_a_coeffs=np.array([1.0]),
        blf_a_const=np.array([0.0]),
        blf_b_ptr=np.array([0, 1], dtype=np.int64),
        blf_b_cols=np.array([2], dtype=np.int64),
        blf_b_coeffs=np.array([1.0]),
        blf_b_const=np.array([0.0]),
        obbt_candidates=np.zeros(0, dtype=np.int64),
    )
    lo = np.array([0.0, 1.0, 1.0, 0.0])
    hi = np.array([10.0, 4.0, 4.0, 100.0])
    live, plo, phi = _rust.propagate_spec_py(**spec, lo=lo, hi=hi, cutoff=2.0, rounds=15)
    assert live
    assert phi[0] <= 2.0 + 1e-6 and phi[3] <= 2.0 + 1e-5
    assert phi[1] <= 2.0 + 1e-4 and phi[2] <= 2.0 + 1e-4
    # Inputs are not mutated, and no cutoff means no cutoff deduction.
    assert hi[0] == 10.0
    live, _, phi = _rust.propagate_spec_py(**spec, lo=lo, hi=hi, cutoff=None, rounds=15)
    assert live and phi[0] == 10.0
    # Cutoff below every feasible objective: the box is proven empty.
    live, _, _ = _rust.propagate_spec_py(**spec, lo=lo + [1.0, 0, 0, 0], hi=hi, cutoff=0.5)
    assert not live
    with pytest.raises(ValueError, match="length n_cols"):
        _rust.propagate_spec_py(**spec, lo=lo[:3], hi=hi)
    with pytest.raises(ValueError, match="NaN"):
        _rust.propagate_spec_py(**spec, lo=lo, hi=hi, cutoff=float("nan"))


def test_binding_replays_the_trees_own_propagation_bit_for_bit():
    """Every observed node's post-propagation box equals ``propagate_spec_py`` applied to
    its pre-propagation box under its cutoff with the tree's round cap; and the observer
    changes nothing (same status / bound / nodes with and without it)."""
    spec = _spec()
    cfg = dict(max_nodes=60, time_limit_s=60.0, propagation_rounds=15)
    recs = []
    observed = _rust.solve_spatial_tree_py(**spec, **cfg, node_observer=recs.append)
    plain = _rust.solve_spatial_tree_py(**spec, **cfg)
    for k in ("status", "incumbent", "bound", "node_count", "n_lp_solves"):
        assert observed[k] == plain[k], (k, observed[k], plain[k])
    assert recs, "the observer never fired"
    n_replayed = n_cutoff = 0
    for r in recs:
        assert {"lo", "hi", "pre_lo", "pre_hi", "cutoff"} <= set(r)
        live, lo, hi = _rust.propagate_spec_py(
            **spec, lo=r["pre_lo"], hi=r["pre_hi"], cutoff=r["cutoff"], rounds=15
        )
        if np.array_equal(r["pre_lo"], r["lo"]) and np.array_equal(r["pre_hi"], r["hi"]):
            # Either nothing to tighten, or the #1995 retry re-solved over the
            # pre-propagation box; both leave the observed box equal to the popped one.
            continue
        assert live
        np.testing.assert_array_equal(lo, r["lo"])
        np.testing.assert_array_equal(hi, r["hi"])
        n_replayed += 1
        n_cutoff += r["cutoff"] is not None
    # Prove the probe fired (CLAUDE.md §6): propagation moved real boxes, under a cutoff.
    assert n_replayed > 0 and n_cutoff > 0, (n_replayed, n_cutoff, len(recs))


def test_observer_has_no_pre_box_when_propagation_is_off():
    spec = _spec()
    recs = []
    _rust.solve_spatial_tree_py(
        **spec, max_nodes=5, time_limit_s=30.0, run_propagation=False, node_observer=recs.append
    )
    assert recs
    assert all("pre_lo" not in r and "lo" in r for r in recs)
