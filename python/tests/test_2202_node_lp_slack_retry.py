"""#2202: an undecided native-kernel node LP is re-solved once from the slack basis.

``DISCOPT_NODE_LP_SLACK_RETRY`` (``SpatialTreeConfig::node_lp_slack_retry``, default
ON, ``=0`` opts out). A child's node LP starts from its ancestor's basis (#1935); on
MINLPLib ``nous1`` that start sends the warm dual into a breakdown and the cold
primal fallback ends ``Numerical`` too, so the node is undecided and branches with
its parent's bound. Under fixed-row permutations the cascade stalled nous1 at the
time limit with ~1,400 undecided nodes, with ``DISCOPT_DUAL_PROOF_PROPAGATION`` on
or off: the macOS/arm64 stall #2202 reported. The captured LPs themselves are pinned
in Rust (``bnb::spatial_kernel::tests::
inherited_basis_numerical_node_lp_is_decided_by_slack_retry_2202``); these tests pin
the solver-level wiring and the class.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import discopt.modeling as dm
import discopt.transformations as dt
import numpy as np
import pytest
from discopt import _rust

sys.path.insert(0, os.path.dirname(__file__))
import serialize_builder_cases as cases  # noqa: E402

FLAG = "DISCOPT_NODE_LP_SLACK_RETRY"
PROP_FLAG = "DISCOPT_KERNEL_UNDECIDED_RETRY"
NOUS1 = Path(__file__).parent / "data" / "issue_1871" / "nous1.nl"
NOUS1_OPT = 1.56707203257682


def _either_or():
    """#1995's case: the ``gdp.simplex`` lowering of a two-disjunct ``either_or``,
    whose objective cutoff collapses node boxes to slivers where the McCormick LP
    ends ``Numerical``."""
    return dt.create_using("gdp.simplex", cases._either_or("select_one")())


@pytest.mark.smoke
def test_the_retry_is_on_by_default_and_zero_opts_out(monkeypatch):
    from discopt.solver import _kernel_node_lp_slack_retry_enabled

    monkeypatch.delenv(FLAG, raising=False)
    assert _kernel_node_lp_slack_retry_enabled()
    for off in ("0", "false", "off"):
        monkeypatch.setenv(FLAG, off)
        assert not _kernel_node_lp_slack_retry_enabled()


@pytest.mark.smoke
def test_the_kernel_default_is_off():
    """The Rust default is the tree before the retry; only the solver turns it on."""
    sig = _rust.solve_spatial_tree_py.__text_signature__ or ""
    assert "node_lp_slack_retry=False" in sig, sig


@pytest.mark.smoke
def test_slack_retry_alone_decides_the_sliver_node(monkeypatch):
    """With #1995's pre-propagation re-solve OFF, the slack-basis re-solve on its own
    decides the sliver node LP and the lowering certifies; with both OFF the legacy
    stall is intact and nothing is counted."""
    monkeypatch.setenv(PROP_FLAG, "0")
    monkeypatch.delenv(FLAG, raising=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = _either_or().solve(time_limit=30)
    stats = r.solver_stats or {}
    assert r.status == "optimal" and r.gap_certified, (r.status, r.node_count, r.bound)
    assert abs(r.objective - 5.0) <= 1e-6
    assert r.bound <= 5.0 + 1e-6
    assert stats.get("tree/prop_retries") == 0.0, stats
    assert stats.get("tree/slack_retry_decided", 0) >= 1, stats  # it fired (§6)
    assert stats["tree/slack_retry_decided"] <= stats["tree/slack_retries"]
    # #2202 also surfaced the kernel's dual-proof counters in ``solver_stats``.
    for k in ("rays", "global_lps", "learned", "tightenings", "fathoms"):
        assert f"tree/dual_proof_{k}" in stats, sorted(stats)

    monkeypatch.setenv(FLAG, "0")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        off = _either_or().solve(time_limit=3)
    assert off.status != "optimal", (off.status, off.node_count)
    assert (off.solver_stats or {}).get("tree/slack_retries") == 0.0


def _permute_rows(kwargs: dict, seed: int) -> dict:
    """The kernel kwargs with the fixed rows permuted by ``seed`` (0 = unchanged): the
    same LP, a different order of floating-point work (#2202's platform class)."""
    k = dict(kwargs)
    if not seed:
        return k
    ptr = np.asarray(k["fixed_row_ptr"])
    cols = np.asarray(k["fixed_cols"])
    co = np.asarray(k["fixed_coeffs"])
    rhs = np.asarray(k["fixed_rhs"])
    perm = np.random.default_rng(seed).permutation(len(rhs))
    nc, ncf, nptr = [], [], [0]
    for q in perm:
        nc.append(cols[ptr[q] : ptr[q + 1]])
        ncf.append(co[ptr[q] : ptr[q + 1]])
        nptr.append(nptr[-1] + ptr[q + 1] - ptr[q])
    k["fixed_row_ptr"] = np.array(nptr, dtype=ptr.dtype)
    k["fixed_cols"] = np.concatenate(nc).astype(cols.dtype)
    k["fixed_coeffs"] = np.concatenate(ncf)
    k["fixed_rhs"] = rhs[perm].copy()
    return k


@pytest.mark.slow
@pytest.mark.correctness
def test_permuted_nous1_has_no_undecided_nodes(monkeypatch):
    """nous1's kernel tree under several fixed-row orders, 500 nodes each: every run
    ends with 0 undecided nodes, and the retry decides at least one node LP somewhere.

    WHICH order triggers the retry depends on the platform's floating-point order (the
    class this test is about): on Linux the unpermuted model and seeds 2, 7, 9 fire at
    this budget; on macOS/arm64 seed 5 never fires within 2,000 nodes while others do
    (#2209 review). So the "probe fired" guard (§6) is over the set, not one seed.
    Without the retry, seed 5 on Linux stalls at the time limit with ~1,400 undecided
    nodes. The kwargs are captured from one ``Model.solve`` and the kernel is then
    called directly; the Python primal hook and pricer are dropped for speed (the
    seeded initial incumbent stays), since the retry lives in the node LP.
    """
    monkeypatch.delenv(FLAG, raising=False)
    orig = _rust.solve_spatial_tree_py
    captured: dict = {}

    def capture(*a, **k):
        if int(k.get("max_nodes", 10**9)) > 1 and not captured:
            captured.update(k)
        return orig(*a, **k)

    monkeypatch.setattr(_rust, "solve_spatial_tree_py", capture)
    dm.from_nl(str(NOUS1)).solve(
        time_limit=120, gap_tolerance=1e-4, deterministic=True, max_nodes=50
    )
    assert captured, "the kernel never ran"
    assert captured.get("node_lp_slack_retry") is True, "the solver must pass the retry"
    runs = []
    for seed in (0, 1, 2, 5, 7, 9):
        k = _permute_rows(captured, seed)
        for drop in ("primal_hook", "incumbent_value", "bound_time_extension_s"):
            k.pop(drop, None)
        k.update(max_nodes=500, time_limit_s=120.0)
        res = orig(**k)
        runs.append((seed, res["n_undecided"], res["n_slack_retry_decided"], res["bound"]))
    assert all(u == 0 for _, u, _, _ in runs), runs
    assert sum(d for _, _, d, _ in runs) >= 1, runs  # the retry fired somewhere (§6)
    assert all(b <= NOUS1_OPT + 1e-6 for _, _, _, b in runs), runs
