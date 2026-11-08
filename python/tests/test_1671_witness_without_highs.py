"""#1671 closed by the in-house engine: the HiGHS-presolve-cycle witness, solved without HiGHS.

#1671 recorded an upstream HiGHS bug: on the #1667 unit-commitment MILP (2904 x 4344,
240 binaries) HiGHS's MIP presolve cycles in ``fastPresolveLoop`` and ignores
``time_limit``; discopt's HiGHS route works around it by switching presolve rule 14
(sparsify) off. The owner's resolution (2026-11-08): the issue is irrelevant once
discopt's own engine solves that problem with no HiGHS at all. This test is that
condition, in a fresh interpreter whose import system refuses ``highspy``: a plain
``m.solve`` must take the native Rust MILP route and certify the known optimum.

Measured 2026-11-08 (highspy blocked, 3 runs): ``optimal`` 450623.7692106857, certified,
29.1-29.6 s, route ``native-milp ... (default; highspy is not installed)``. Before the
highs-parity work the Rust route found no incumbent in 120 s (P0.7,
``docs/dev/data/highs-parity/uc-timelimit-fix/``). The 120 s limit is a backstop, sized
well above the measured time (CLAUDE.md, slow-lane rules), not the quality bar.
``test_1671_highs_presolve_cycle_canary.py`` keeps recording the upstream bug itself.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

HERE = Path(__file__).parent
UC_OPT = 450623.7692106868

_BLOCK = textwrap.dedent(
    """\
    import sys
    class _NoHighs:
        def find_spec(self, name, path=None, target=None):
            if name.partition(".")[0] == "highspy":
                raise ModuleNotFoundError(f"No module named {name!r}", name=name)
            return None
    sys.meta_path.insert(0, _NoHighs())
    """
)


def _uc_script(time_limit: float) -> str:
    src = (HERE / "test_1667_highs_sparsify_presolve_cycle.py").read_text()
    start = 'UC_SCRIPT = textwrap.dedent(\n    """\\\n'
    i = src.index(start) + len(start)
    j = src.index('    """\n)', i)
    body = textwrap.dedent(src[i:j])
    assert "res = m.solve(time_limit=20)" in body, "the #1667 witness script changed"
    body = body.replace("res = m.solve(time_limit=20)", f"res = m.solve(time_limit={time_limit})")
    return (
        _BLOCK
        + body
        + "import sys as _s\n"
        + "print('ROUTE', res.algorithm_route, flush=True)\n"
        + "print('CERT', bool(res.gap_certified), 'HIGHSPY', 'highspy' in _s.modules"
        + " and _s.modules['highspy'] is not None, flush=True)\n"
    )


@pytest.mark.slow
@pytest.mark.correctness
def test_the_1671_witness_is_certified_without_highs():
    p = subprocess.run(
        [sys.executable, "-u", "-c", _uc_script(120.0)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert p.returncode == 0, p.stderr[-3000:]
    out = {ln.split()[0]: ln.split(None, 1)[1] for ln in p.stdout.splitlines() if ln[:5] in (
        "RESUL", "ROUTE", "CERT ")}  # fmt: skip
    status, obj, bound, _wall = out["RESULT"].split()
    assert status == "optimal", p.stdout[-2000:]
    assert abs(float(obj) - UC_OPT) <= 1e-6 * (1.0 + UC_OPT), obj
    assert float(bound) <= UC_OPT + 1e-6 * (1.0 + UC_OPT), bound
    assert out["CERT"].split()[0] == "True" and out["CERT"].split()[2] == "False", out["CERT"]
    assert out["ROUTE"].startswith("native-milp") and "highspy is not installed" in out["ROUTE"]
