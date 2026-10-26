"""#1962: per-node separation of convex quadratic forms in the native spatial kernel.

The kernel relaxes every product by its own McCormick envelope, so a convex
quadratic form ``F(w) = sum a_k w_k = x'Qx + l'x + c`` (``Q`` PSD) loses its
convexity: nvs23 stopped at the 100k-node limit with its bound at -1254.8 against
an optimum of -1125.2. ``DISCOPT_KERNEL_CONVEX_CUTS`` adds the outer-approximation
tangent of every certified convex form at the node LP point, re-solves warm, and
hands the binding cuts to the children
(``crates/discopt-core/src/bnb/convex_quad_cuts.rs``).

``DISCOPT_KERNEL_INT_PROBE`` is the primal half nvs24 needed: once the cuts put its
bound on the optimum, the incumbent stayed at the seed because a pure-integer
search reaches a McCormick-tight vertex only once branching has collapsed every
operand's box. The probe fixes the integer columns of an integral LP point and
re-solves.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import discopt.modeling as dm
import pytest
from discopt._relax.spatial_producer import solve_with_native_kernel

DATA = Path(__file__).parent / "data" / "minlplib"


def _iqp_model(integer: bool = True):
    """``min x0^2 + x1^2 + x2^2 + x0*x1 + x1*x2 - 7x0 - 8x1 - 5x2`` over ``[0, 10]``
    with a nonconvex side constraint, so the solve is spatial. ``Model.solve``
    sends the integer version to the exact MILP route, so the end-to-end test uses
    the continuous one."""
    m = dm.Model("iqp1962")
    if integer:
        x = [m.integer(f"x{i}", lb=0, ub=10) for i in range(3)]
    else:
        x = [m.continuous(f"x{i}", lb=0, ub=10) for i in range(3)]
    m.minimize(
        x[0] * x[0]
        + x[1] * x[1]
        + x[2] * x[2]
        + x[0] * x[1]
        + x[1] * x[2]
        - 7 * x[0]
        - 8 * x[1]
        - 5 * x[2]
    )
    m.subject_to(x[0] + x[1] + x[2] <= 8)
    m.subject_to(x[0] * x[2] - x[1] >= 1)  # nonconvex
    return m


def _iqp_oracle() -> float:
    best = float("inf")
    for a, b, c in itertools.product(range(11), repeat=3):
        if a + b + c <= 8 and a * c - b >= 1:
            best = min(best, a * a + b * b + c * c + a * b + b * c - 7 * a - 8 * b - 5 * c)
    return best


def test_kernel_cut_loop_tightens_root_and_keeps_the_optimum():
    """The kernel with the loop certifies the same optimum, from a tighter root,
    and reports that the separator ran (CLAUDE.md section 6)."""
    opt = _iqp_oracle()
    off = solve_with_native_kernel(_iqp_model())
    on = solve_with_native_kernel(_iqp_model(), cut_rounds_root=30, cut_rounds_node=3)
    assert off is not None and on is not None
    assert off["status"] == on["status"] == "optimal"
    assert off["n_cut_forms"] == off["n_cut_rounds"] == off["n_cuts_added"] == 0
    assert on["n_cut_forms"] >= 1
    assert on["n_cut_rounds"] > 0 and on["n_cuts_added"] > 0
    sign, offset = on["meta_obj_sense_sign"], on["meta_obj_offset"]
    for r in (off, on):
        assert sign * (r["incumbent"] + offset) == pytest.approx(opt, abs=1e-6)
        assert sign * (r["bound"] + offset) <= opt + 1e-6
        assert sign * (r["root_bound"] + offset) <= opt + 1e-6
    assert on["root_bound"] > off["root_bound"] + 1.0, (on["root_bound"], off["root_bound"])
    assert on["node_count"] <= off["node_count"]


def test_flags_reach_the_kernel_and_opt_out(monkeypatch):
    """``Model.solve`` passes both flags through; ``=0`` turns the separator off.
    The separator is on by default (graduated in #1962)."""
    monkeypatch.setenv("DISCOPT_KERNEL_CONVEX_CUTS", "0")
    monkeypatch.setenv("DISCOPT_KERNEL_INT_PROBE", "0")
    r0 = _iqp_model(integer=False).solve(time_limit=60)
    s0 = r0.solver_stats or {}
    assert s0.get("tree/nodes") is not None, "solve did not take the native kernel"
    assert s0.get("tree/cut_forms") == 0.0 and s0.get("tree/cut_rounds") == 0.0
    assert s0.get("tree/int_probes") == 0.0

    monkeypatch.delenv("DISCOPT_KERNEL_CONVEX_CUTS", raising=False)
    monkeypatch.setenv("DISCOPT_KERNEL_INT_PROBE", "1")
    r1 = _iqp_model(integer=False).solve(time_limit=60)
    s1 = r1.solver_stats or {}
    assert s1.get("tree/nodes") is not None, "solve did not take the native kernel"
    assert s1.get("tree/cut_forms", 0.0) >= 1.0
    assert s1.get("tree/cut_rounds", 0.0) > 0.0
    assert r0.status == r1.status == "optimal"
    assert r1.objective == pytest.approx(r0.objective, abs=1e-5)
    for r in (r0, r1):
        assert r.bound <= r.objective + 1e-6


@pytest.mark.slow
@pytest.mark.correctness
@pytest.mark.parametrize("name,opt", [("nvs23", -1125.2), ("nvs24", -1033.2)])
def test_nvs_certifies_with_cuts_and_probe(monkeypatch, name, opt):
    """The issue's targets: on ``main`` nvs23 ends at the node limit (bound
    -1254.8) and nvs24 at the time limit (-1614.6). With both flags each
    certifies in a few hundred nodes."""
    monkeypatch.setenv("DISCOPT_KERNEL_CONVEX_CUTS", "1")
    monkeypatch.setenv("DISCOPT_KERNEL_INT_PROBE", "1")
    r = dm.from_nl(str(DATA / f"{name}.nl")).solve(time_limit=120)
    assert r.status == "optimal", (r.status, r.objective, r.bound, r.node_count)
    assert r.objective == pytest.approx(opt, abs=1e-4)
    assert r.bound <= opt + 1e-6
    assert (r.solver_stats or {}).get("tree/cut_rounds", 0.0) > 0.0
