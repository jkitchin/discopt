"""End-to-end Tier-3 certificate from a real branch-and-bound solve.

Solves a real MINLPLib instance (``nvs03``, a convex integer NLP that runs the
recorded NLP-BB path) with ``emit_certificate=True``, builds a Tier-3 ``bnb``
certificate, and verifies it: the leaf boxes cover the root box and the reported
dual bound is a valid global lower bound. Also exercises the untrusted per-leaf
re-derivation, bound-neutrality, and tamper rejection. A real solve runs, so these
are ``slow``.

(``nvs03`` is used rather than a synthetic quadratic because the solver routes pure
quadratic models to a specialized QP/MIQP solver that does not build a recorded
spatial-B&B tree; ``nvs03`` reliably runs a recorded path and is exactly-rational
so the checker can verify it.)
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
from discopt.certificate import (
    CertificateError,
    build_bnb_certificate,
    check_certificate,
)
from discopt.modeling.core import from_nl

_NL = Path(__file__).parent / "data" / "minlplib_nl" / "nvs03.nl"


def _solve(emit=True):
    if not _NL.exists():
        pytest.skip("nvs03.nl not in corpus")
    m = from_nl(str(_NL))
    r = m.solve(emit_certificate=emit, gap_tolerance=1e-3, time_limit=30, max_nodes=2000)
    return m, r


@pytest.fixture(scope="module")
def solved():
    m, r = _solve(emit=True)
    if getattr(r, "bnb_tree", None) is None:
        pytest.skip("nvs03 did not run a recorded B&B path on this build")
    return m, r


@pytest.mark.slow
def test_bnb_certificate_accepts_real_solve(solved):
    m, r = solved
    cert = build_bnb_certificate(m, r)
    assert cert["certificate"]["tier"] == "bnb"
    ok, reason = check_certificate(cert)
    assert ok, reason
    assert "cover the root box" in reason


@pytest.mark.slow
def test_bnb_untrusted_leaf_rederivation(solved):
    """The checker re-derives leaf bounds by rebuilding the McCormick LP and
    verifying the emitted dual -- trusting neither the solver's bound nor the
    emitted LP. A tampered dual is rejected."""
    m, r = solved
    cert = build_bnb_certificate(m, r, untrusted=True)
    n_ut = cert["certificate"]["tree"].get("untrusted_leaves", 0)
    if n_ut == 0:
        pytest.skip("no quadratic-fragment leaves to re-derive on this instance")
    ok, reason = check_certificate(cert)
    assert ok, reason
    assert "re-derived (untrusted)" in reason

    tampered = copy.deepcopy(cert)
    leaf = next(n for n in tampered["certificate"]["tree"]["nodes"] if "untrusted_dual" in n)
    leaf["untrusted_dual"][0] = [999, 1]
    assert not check_certificate(tampered)[0]


@pytest.mark.slow
def test_bnb_recording_is_bound_neutral():
    _m1, off = _solve(emit=False)
    _m2, on = _solve(emit=True)
    assert off.node_count == on.node_count
    assert abs(off.objective - on.objective) < 1e-9
    assert on.bnb_tree is not None and off.bnb_tree is None


@pytest.mark.slow
def test_bnb_rejects_inflated_dual_bound(solved):
    m, r = solved
    cert = copy.deepcopy(build_bnb_certificate(m, r))
    cert["certificate"]["dualBound"] = [10**9, 1]  # claim a bound above the leaves
    ok, reason = check_certificate(cert)
    assert not ok and "leaf bound" in reason.lower()


@pytest.mark.slow
def test_bnb_rejects_broken_covering(solved):
    m, r = solved
    cert = copy.deepcopy(build_bnb_certificate(m, r))
    nodes = cert["certificate"]["tree"]["nodes"]
    internal = {n["parent"] for n in nodes if n["parent"] is not None}
    leaf = next(n for n in nodes if n["id"] not in internal)
    leaf["lb"][0] = [999, 1]  # shove a leaf box off its split -> covering gap
    assert not check_certificate(cert)[0]


@pytest.mark.slow
def test_bnb_emitter_refuses_without_recording():
    _m, r = _solve(emit=False)
    m = from_nl(str(_NL))
    with pytest.raises(CertificateError):
        build_bnb_certificate(m, r)


@pytest.mark.slow
def test_cli_emits_and_checks_bnb_certificate(tmp_path, monkeypatch):
    """`discopt solve --emit-certificate` on a real .nl -> a Tier-3 bnb cert that
    `discopt cert-check` accepts."""
    import json
    import sys

    if not _NL.exists():
        pytest.skip("nvs03.nl not in corpus")
    from discopt.cli import main

    def _run(argv):
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit) as exc:
            main()
        return exc.value.code

    _run(
        [
            "discopt",
            "solve",
            str(_NL),
            "--emit-certificate",
            "--out-dir",
            str(tmp_path),
            "--quiet",
            "--gap",
            "1e-3",
        ]
    )
    cert_path = tmp_path / "nvs03.cert.json"
    assert cert_path.exists()
    cert = json.loads(cert_path.read_text())
    if cert["certificate"]["tier"] != "bnb":
        pytest.skip("nvs03 did not emit a bnb-tier certificate on this build")
    assert _run(["discopt", "cert-check", str(cert_path)]) == 0
