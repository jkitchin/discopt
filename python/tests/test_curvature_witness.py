"""Indefinite-point witnesses short-circuit curvature proofs that must abstain.

``uniform_relax._Builder._curvature_cert`` proves box convexity/concavity of a
lifted multivariate node with ``milp_relaxation._multivar_box_curvature`` (interval
Hessian + Gershgorin on a refined partition). On GDPlib ex_633 (hull) every one of
375 proofs abstained, and 321 of them had a Hessian that is provably indefinite at
the box midpoint. On a box containing such a point the proof cannot succeed: every
partition has a sub-box holding the point, and that sub-box's Gershgorin bounds
bracket the point Hessian's spectrum. ``_certified_indefinite_at`` certifies the
point; ``_curvature_cert`` then returns the proof's verdict (``None``) without
running it.

Pinned here (CLAUDE.md 5, bound-neutral; 6, prove the probe fired):

* the witness test is sound: indefinite where the Hessian is indefinite, never on
  a convex, concave, or flat (linear) node;
* a skipped proof is one that abstains: every witness skip is re-proven and
  returns ``None``;
* per-call verdicts with the witness are identical to the always-prove path, on a
  sequence of shrinking boxes, while fewer proofs run.
"""

from __future__ import annotations

import discopt._relax.milp_relaxation as MR
import discopt._relax.uniform_relax as UR
import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.milp_relaxation import _build_convexity_box, _certified_indefinite_at
from discopt._relax.uniform_relax import build_uniform_relaxation
from discopt.modeling.core import Model

pytestmark = pytest.mark.unit


def _two_var(lo=-1.0, hi=2.0):
    m = Model("witness")
    x = m.continuous("x", lb=lo, ub=hi)
    y = m.continuous("y", lb=lo, ub=hi)
    return m, x, y


def _indef(expr, m, point):
    lb = np.array([v.lb for v in m._variables], dtype=np.float64).reshape(-1)
    ub = np.array([v.ub for v in m._variables], dtype=np.float64).reshape(-1)
    box = _build_convexity_box(m, lb, ub)
    return _certified_indefinite_at(expr, m, [0, 1], np.asarray(point, dtype=np.float64), box)


def test_witness_is_found_only_where_the_hessian_is_indefinite():
    m, x, y = _two_var()
    checked = 0
    for expr, point, want in [
        (x * y, [0.5, 0.5], True),  # [[0, 1], [1, 0]]
        (dm.exp(x * y), [0.5, 0.5], True),
        (x**2 - y**2, [0.3, -0.7], True),
        (x**2 + y**2, [0.5, 0.5], False),  # convex
        (-(x**2) - 3 * y**2, [0.5, 0.5], False),  # concave
        (x**2 + 2 * x * y + y**2, [0.5, 0.5], False),  # PSD, singular
        (2.0 * x + 3.0 * y, [0.5, 0.5], False),  # flat: zero Hessian
        (x * y + 1e-12 * x**2, [0.0, 0.0], True),
        (1e-10 * x * y, [0.5, 0.5], False),  # inside the 2*tol margin
    ]:
        assert _indef(expr, m, point) is want, (expr, point)
        checked += 1
    assert checked == 9


def test_witness_test_restores_the_box():
    m, x, y = _two_var()
    lb = np.array([-1.0, -1.0])
    ub = np.array([2.0, 2.0])
    box = _build_convexity_box(m, lb, ub)
    before = {v: (np.copy(box[v].lo), np.copy(box[v].hi)) for v in box}
    assert _certified_indefinite_at(x * y, m, [0, 1], np.array([0.5, 0.5]), box)
    for v, (lo, hi) in before.items():
        np.testing.assert_array_equal(box[v].lo, lo)
        np.testing.assert_array_equal(box[v].hi, hi)


def _model():
    m = Model("witness_relax")
    x = m.continuous("x", lb=-1, ub=2)
    y = m.continuous("y", lb=-1, ub=2)
    z = m.continuous("z", lb=0.5, ub=3)
    m.subject_to(dm.exp(x * y) + z <= 5.0)
    m.subject_to(dm.log(z) * x + y**2 <= 3.0)
    m.minimize(x + y + z)
    return m


def _run(monkeypatch, witness: bool):
    """Build over boxes halving around their centre; record each verdict."""
    monkeypatch.setattr(UR, "_DISCOPT_CURV_WITNESS", witness)
    verdicts: list = []
    proofs = {"n": 0, "skipped_rechecked": 0}
    real_cert = UR._Builder._curvature_cert
    real_proof = MR._multivar_box_curvature

    def proof(*a, **k):
        proofs["n"] += 1
        return real_proof(*a, **k)

    def cert(self, node, idxs, flat_lb, flat_ub, cbox):
        before = proofs["n"]
        v = real_cert(self, node, idxs, flat_lb, flat_ub, cbox)
        verdicts.append((tuple(idxs), v))
        sel = np.asarray(idxs)
        lo, hi = flat_lb[sel], flat_ub[sel]
        witnesses = self._analysis.hessian_witness.get(id(node), ())
        if proofs["n"] == before and any(np.all(lo <= w) and np.all(w <= hi) for w in witnesses):
            # A witness skip: the real proof must abstain on this box.
            assert real_proof(self._expr(node), self.model, idxs, flat_lb, flat_ub, cbox) is None
            proofs["skipped_rechecked"] += 1
        return v

    monkeypatch.setattr(MR, "_multivar_box_curvature", proof)
    monkeypatch.setattr(UR._Builder, "_curvature_cert", cert)
    m = _model()
    lb = np.array([-1.0, -1.0, 0.5])
    ub = np.array([2.0, 2.0, 3.0])
    for _ in range(6):
        build_uniform_relaxation(m, box=(lb.copy(), ub.copy()))
        w = ub - lb
        lb, ub = lb + w / 4, ub - w / 4
    return verdicts, proofs


def test_witness_skips_only_abstaining_proofs_and_keeps_every_verdict(monkeypatch):
    legacy, legacy_proofs = _run(monkeypatch, witness=False)
    monkeypatch.undo()
    fast, fast_proofs = _run(monkeypatch, witness=True)
    assert len(legacy) == 24
    assert fast == legacy
    assert legacy_proofs["skipped_rechecked"] == 0
    assert fast_proofs["skipped_rechecked"] > 0, "no witness skip happened -- nothing was tested"
    assert fast_proofs["n"] < legacy_proofs["n"]
