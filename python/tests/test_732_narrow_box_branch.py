"""#732 Stage 1 — narrow-box branch-instead-of-taint (``DISCOPT_NARROW_BOX_BRANCH``).

A nonconvex node whose relaxation LP fails numerically on a narrow/pinned box is
normally fathomed *non-rigorously*, which taints the tree's certified dual bound
and forces it to be discarded (ex1252: the internal frontier reaches the optimum
yet the reported bound collapses to ~0). With the flag ON, a *branchable* failed
node is kept OPEN at its rigorous parent-inherited bound and bisected instead, so
its children re-solve on narrower, better-conditioned boxes.

Soundness is the point of the flag: it must never report a dual bound above the
true optimum (a false certificate), and it must be byte-identical when OFF.
"""

from __future__ import annotations

import os

import discopt.modeling as dm
import pytest

_EX1252 = "python/tests/data/minlplib/ex1252.nl"
_EX1252_OPT = 128893.741


def _solve(nl_or_model, *, nbb: str, rlt: str = "0", time_limit: float = 60.0):
    prev_nbb = os.environ.get("DISCOPT_NARROW_BOX_BRANCH")
    prev_rlt = os.environ.get("DISCOPT_MULTILINEAR_COUPLING_RLT")
    os.environ["DISCOPT_NARROW_BOX_BRANCH"] = nbb
    os.environ["DISCOPT_MULTILINEAR_COUPLING_RLT"] = rlt
    try:
        model = dm.from_nl(nl_or_model) if isinstance(nl_or_model, str) else nl_or_model
        return model.solve(time_limit=time_limit)
    finally:
        for k, v in (
            ("DISCOPT_NARROW_BOX_BRANCH", prev_nbb),
            ("DISCOPT_MULTILINEAR_COUPLING_RLT", prev_rlt),
        ):
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _tiny_nonconvex():
    """A small well-conditioned nonconvex model with no narrow-box LP failures,
    so the flag is inert on it (ON must equal OFF exactly)."""
    m = dm.Model()
    x = m.continuous("x", lb=-2.0, ub=2.0)
    y = m.continuous("y", lb=-2.0, ub=2.0)
    m.subject_to(x * y >= 1.0)
    m.minimize(x * y + x * x + y * y)
    return m


def test_flag_off_and_on_byte_identical_on_clean_model():
    """No narrow-box failure -> the flag changes nothing: same node_count and
    objective ON vs OFF. Guards the OFF path's inertness."""
    off = _solve(_tiny_nonconvex(), nbb="0", time_limit=30)
    on = _solve(_tiny_nonconvex(), nbb="1", time_limit=30)
    assert off.node_count == on.node_count
    assert (off.objective is None) == (on.objective is None)
    if off.objective is not None:
        assert off.objective == pytest.approx(on.objective, abs=1e-6, rel=1e-6)
    assert (off.bound is None) == (on.bound is None)
    if off.bound is not None and on.bound is not None:
        assert off.bound == pytest.approx(on.bound, abs=1e-6, rel=1e-6)


@pytest.mark.slow
@pytest.mark.skipif(
    not os.path.exists(_EX1252),
    reason="ex1252.nl corpus file not present",
)
def test_ex1252_untaints_soundly():
    """Before #732 Stage 1-A, ex1252 (coupling-RLT ON) reports a discarded ~0 dual
    bound because narrow-box LP failures taint the tree. With the flag ON the bound
    is a real, rigorous value again — and must remain SOUND (never above the true
    optimum). This is the fails-before / passes-after regression."""
    r = _solve(_EX1252, nbb="1", rlt="1", time_limit=90)
    tol = 1e-3 * (1 + abs(_EX1252_OPT))

    # Un-tainted: the reported dual bound is a real positive number, not the
    # ~0 / discarded sentinel the taint path produced before the fix.
    assert r.bound is not None, "flag ON must surface a rigorous dual bound"
    assert r.bound > 1.0, f"bound {r.bound} still looks discarded (taint not removed)"

    # SOUND (non-negotiable): the dual bound never exceeds the true optimum.
    assert r.bound <= _EX1252_OPT + tol, (
        f"FALSE CERTIFICATE: bound {r.bound} > oracle {_EX1252_OPT}"
    )

    # Any incumbent found is a genuine feasible point (>= the optimum).
    if r.objective is not None:
        assert r.objective >= _EX1252_OPT - tol, (
            f"incumbent {r.objective} below oracle {_EX1252_OPT}"
        )
