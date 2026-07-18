"""End-to-end Tier-3 certificate from a real spatial branch-and-bound solve.

Solves a nonconvex bilinear model with ``emit_certificate=True`` (so the Rust
recorder stashes the tree on the result), builds a Tier-3 ``bnb`` certificate, and
verifies it: the leaf boxes cover the root box and the reported dual bound is a
valid global lower bound. Also checks bound-neutrality (recording changes no
answer) and tamper rejection. A real solve runs, so these are ``slow``.
"""

from __future__ import annotations

import discopt.modeling as dm
import pytest
from discopt.certificate import build_bnb_certificate, check_certificate


def _nonconvex_chain(n=4):
    """min sum x_i x_{i+1} s.t. sum x_i = 2, x in [-1,2]^n -- loose McCormick root
    gap, so spatial B&B branches (multi-node tree)."""
    m = dm.Model()
    xs = [m.continuous(f"x{i}", lb=-1, ub=2) for i in range(n)]
    m.subject_to(sum(xs) == 2, name="sum")
    m.minimize(sum(xs[i] * xs[i + 1] for i in range(n - 1)))
    return m


@pytest.mark.slow
def test_bnb_certificate_accepts_real_solve():
    m = _nonconvex_chain()
    r = m.solve(emit_certificate=True, gap_tolerance=1e-3, time_limit=60, max_nodes=500)
    assert r.bnb_tree is not None and len(r.bnb_tree) >= 1
    cert = build_bnb_certificate(m, r)
    assert cert["certificate"]["tier"] == "bnb"
    ok, reason = check_certificate(cert)
    assert ok, reason
    assert "cover the root box" in reason


@pytest.mark.slow
def test_bnb_recording_is_bound_neutral():
    m1 = _nonconvex_chain()
    off = m1.solve(gap_tolerance=1e-3, time_limit=60, max_nodes=500)
    m2 = _nonconvex_chain()
    on = m2.solve(emit_certificate=True, gap_tolerance=1e-3, time_limit=60, max_nodes=500)
    assert off.node_count == on.node_count
    assert abs(off.objective - on.objective) < 1e-9
    assert on.bnb_tree is not None and off.bnb_tree is None


@pytest.mark.slow
def test_bnb_rejects_inflated_dual_bound():
    m = _nonconvex_chain()
    r = m.solve(emit_certificate=True, gap_tolerance=1e-3, time_limit=60, max_nodes=500)
    cert = build_bnb_certificate(m, r)
    cert["certificate"]["dualBound"] = [10**6, 1]  # claim a bound above the leaves
    ok, reason = check_certificate(cert)
    assert not ok and "leaf bound" in reason.lower()


@pytest.mark.slow
def test_bnb_rejects_broken_covering():
    m = _nonconvex_chain()
    r = m.solve(emit_certificate=True, gap_tolerance=1e-3, time_limit=60, max_nodes=500)
    cert = build_bnb_certificate(m, r)
    nodes = cert["certificate"]["tree"]["nodes"]
    internal = {n["parent"] for n in nodes if n["parent"] is not None}
    leaf = next(n for n in nodes if n["id"] not in internal)
    leaf["lb"][0] = [999, 1]  # shove a leaf box off its split -> covering gap
    ok, _ = check_certificate(cert)
    assert not ok


@pytest.mark.slow
def test_bnb_emitter_refuses_without_recording():
    from discopt.certificate import CertificateError

    m = _nonconvex_chain()
    r = m.solve(gap_tolerance=1e-3, time_limit=60, max_nodes=500)  # no emit_certificate
    with pytest.raises(CertificateError):
        build_bnb_certificate(m, r)
