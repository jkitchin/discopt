"""Regression: LR-2 flags certify nvs09 at (near) the root; OFF they do not.

Task ``cert:LR-2`` (``docs/dev/lever-a-root-tightness-plan.md``). nvs09's objective
is ``Σ (ln(xᵢ-2))² + (ln(10-xᵢ))² − (Πxᵢ)^0.2`` over integer ``xᵢ ∈ [3,9]``, optimum
−43.134. The default relaxation composes the ``(ln)²`` terms (each square floats to
0) and recursive-McCormicks the product, leaving a ~5-unit dual gap that the tree
cannot close in the budget. With ``DISCOPT_UNIVARIATE_ENVELOPE`` (exact 1-D hull of
the univariate composites) — optionally plus ``DISCOPT_LOG_MONOMIAL`` (log-space
product) — the root LP is tight enough to certify in a handful of nodes.

These tests fail on prior main (no flags) and pass with the flags, and assert the
certificate is sound (bound ≤ optimum, |bound − optimum| within tolerance).
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_NL = _REPO / "python" / "tests" / "data" / "minlplib_nl" / "nvs09.nl"
_OPT = -43.134
_TOL = 1e-4 * (1.0 + abs(_OPT))  # 4.4e-3 root-certificate budget (§1.1)


def _solve_subprocess(env_extra: dict[str, str], time_limit: int = 40) -> dict:
    """Solve nvs09 in a fresh interpreter (flags are read at import/build time)."""
    script = textwrap.dedent(
        f"""
        import json
        from discopt.modeling import from_nl
        m = from_nl({str(_NL)!r})
        res = m.solve(time_limit={time_limit})
        print("RESULT " + json.dumps({{
            "status": str(getattr(res, "status", None)),
            "objective": getattr(res, "objective", None),
            "bound": getattr(res, "bound", getattr(res, "dual_bound", None)),
            "nodes": getattr(res, "node_count", getattr(res, "nodes", None)),
        }}))
        """
    )
    env = dict(os.environ)
    env.update(
        {
            "JAX_PLATFORMS": "cpu",
            "JAX_ENABLE_X64": "1",
            "PYTHONPATH": str(_REPO / "python"),
        }
    )
    env.update(env_extra)
    out = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=time_limit + 120,
    )
    line = next((ln for ln in out.stdout.splitlines() if ln.startswith("RESULT ")), None)
    assert line is not None, (
        f"no RESULT line.\nSTDOUT:\n{out.stdout}\nSTDERR:\n{out.stderr[-2000:]}"
    )
    import json

    return json.loads(line[len("RESULT ") :])


def _status_is_optimal(status: str) -> bool:
    return "optimal" in status.lower()


@pytest.mark.slow
def test_nvs09_certifies_with_univariate_envelope_flag():
    r = _solve_subprocess({"DISCOPT_UNIVARIATE_ENVELOPE": "1"})
    assert _status_is_optimal(r["status"]), f"flag ON did not certify: {r}"
    assert r["bound"] is not None
    # sound certificate: bound ≤ optimum and within the root-certificate budget
    assert r["bound"] <= _OPT + 1e-6, f"bound {r['bound']} crossed optimum {_OPT}"
    assert abs(r["bound"] - _OPT) <= _TOL, f"bound {r['bound']} not within {_TOL} of {_OPT}"


@pytest.mark.slow
def test_nvs09_certifies_with_both_flags():
    r = _solve_subprocess({"DISCOPT_UNIVARIATE_ENVELOPE": "1", "DISCOPT_LOG_MONOMIAL": "1"})
    assert _status_is_optimal(r["status"]), f"both flags did not certify: {r}"
    assert r["bound"] <= _OPT + 1e-6
    assert abs(r["bound"] - _OPT) <= _TOL


@pytest.mark.slow
def test_nvs09_does_not_certify_without_flags():
    # The guard the regression protects: default (flags OFF) leaves a real dual gap.
    r = _solve_subprocess({"DISCOPT_UNIVARIATE_ENVELOPE": "0", "DISCOPT_LOG_MONOMIAL": "0"})
    assert not _status_is_optimal(r["status"]), f"unexpectedly certified without flags: {r}"
    # and the OFF bound is materially looser than the optimum (the gap the flags close)
    assert r["bound"] is not None and r["bound"] < _OPT - 1.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-m", "slow"]))
